using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace GameSecretaryApp;

public sealed class BackendManager
{
    public static BackendManager Instance { get; } = new BackendManager();

    private Process? _proc;
    private string? _repoRoot;
    private IntPtr _job;

    public string? RepoRoot => _repoRoot;

    public string ApiHost { get; } = "127.0.0.1";
    public int ApiPort { get; } = 8000;

    public string ApiBase => $"http://{ApiHost}:{ApiPort}/api/v1";
    public string DashboardUrl => $"http://{ApiHost}:{ApiPort}/dashboard.html?v={Uri.EscapeDataString(DateTime.UtcNow.Ticks.ToString())}";

    public string LogsDir
    {
        get
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(_repoRoot))
                {
                    return Path.Combine(_repoRoot, "logs");
                }
            }
            catch
            {
            }
            return Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "GameSecretaryApp",
                "logs"
            );
        }
    }

    public string LocalVlmModel { get; } = "Qwen/Qwen3-VL-8B-Instruct";

    private BackendManager() { }

    private static class Win32Job
    {
        private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
        private const int JobObjectExtendedLimitInformation = 9;

        [StructLayout(LayoutKind.Sequential)]
        private struct IO_COUNTERS
        {
            public ulong ReadOperationCount;
            public ulong WriteOperationCount;
            public ulong OtherOperationCount;
            public ulong ReadTransferCount;
            public ulong WriteTransferCount;
            public ulong OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public long Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string? lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetInformationJobObject(IntPtr hJob, int JobObjectInformationClass, IntPtr lpJobObjectInformation, uint cbJobObjectInformationLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr hObject);

        public static IntPtr EnsureJob(ref IntPtr job)
        {
            if (job != IntPtr.Zero) return job;
            var h = CreateJobObject(IntPtr.Zero, null);
            if (h == IntPtr.Zero) return IntPtr.Zero;
            try
            {
                var info = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                var len = (uint)Marshal.SizeOf<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
                var ptr = Marshal.AllocHGlobal((int)len);
                try
                {
                    Marshal.StructureToPtr(info, ptr, false);
                    if (!SetInformationJobObject(h, JobObjectExtendedLimitInformation, ptr, len))
                    {
                        try { CloseHandle(h); } catch { }
                        return IntPtr.Zero;
                    }
                }
                finally
                {
                    try { Marshal.FreeHGlobal(ptr); } catch { }
                }
            }
            catch
            {
                try { CloseHandle(h); } catch { }
                return IntPtr.Zero;
            }
            job = h;
            return h;
        }

        public static void Assign(IntPtr job, Process p)
        {
            try
            {
                if (job == IntPtr.Zero) return;
                AssignProcessToJobObject(job, p.Handle);
            }
            catch
            {
            }
        }

        public static void Close(ref IntPtr job)
        {
            try
            {
                if (job == IntPtr.Zero) return;
                CloseHandle(job);
            }
            catch
            {
            }
            job = IntPtr.Zero;
        }
    }

    public async Task StartAsync()
    {
        if (_proc is { HasExited: false })
        {
            return;
        }

        _repoRoot = FindRepoRoot();
        if (string.IsNullOrWhiteSpace(_repoRoot))
        {
            throw new InvalidOperationException("Cannot locate repo root (scripts/run_backend.py not found). Run the app from within the repo.");
        }

        if (TcpListening(ApiHost, ApiPort, timeoutMs: 400))
        {
            // A backend (possibly older code/env) is already holding the port.
            // Restart to ensure we use the intended venv/env vars and endpoints.
            TryKillTcpListener(ApiPort);
            await Task.Delay(400);
        }

        Directory.CreateDirectory(Path.Combine(_repoRoot, "logs"));
        var outLog = Path.Combine(_repoRoot, "logs", "backend.out.log");
        var errLog = Path.Combine(_repoRoot, "logs", "backend.err.log");
        var statePath = Path.Combine(_repoRoot, "logs", "backend.state.json");

        // Clear stale agent logs so dashboard doesn't show legacy main.py/ADB traceback
        try { File.WriteAllText(Path.Combine(_repoRoot, "logs", "agent.out.log"), "", Encoding.UTF8); } catch { }
        try { File.WriteAllText(Path.Combine(_repoRoot, "logs", "agent.err.log"), "", Encoding.UTF8); } catch { }

        var venvPy = Path.Combine(_repoRoot, "venv311", "Scripts", "python.exe");
        var stockPy = @"D:\Project\Stock\venv311\Scripts\python.exe";
        var envPy = Environment.GetEnvironmentVariable("GAMESECRETARY_PYTHON") ?? "";
        var py = !string.IsNullOrWhiteSpace(envPy) && File.Exists(envPy)
            ? envPy
            : (File.Exists(venvPy) ? venvPy : (File.Exists(stockPy) ? stockPy : "py"));
        var script = Path.Combine(_repoRoot, "scripts", "run_backend.py");

        var psi = new ProcessStartInfo
        {
            FileName = py,
            WorkingDirectory = _repoRoot,
            CreateNoWindow = true,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        psi.ArgumentList.Add(script);

        try
        {
            psi.Environment["PYTHONUTF8"] = "1";

            // Avoid deprecated Transformers cache env var if user set it globally
            try { psi.Environment.Remove("TRANSFORMERS_CACHE"); } catch { }

            // Force local_vlm only (Qwen3-VL-8B)
            psi.Environment["LOCAL_VLM_MODEL"] = LocalVlmModel;
            var dev = Environment.GetEnvironmentVariable("LOCAL_VLM_DEVICE") ?? "";
            psi.Environment["LOCAL_VLM_DEVICE"] = string.IsNullOrWhiteSpace(dev) ? "cuda" : dev;
            psi.Environment["MODELS_DIR"] = @"D:\Project\ml_cache\models";
            psi.Environment["LOCAL_VLM_MODELS_DIR"] = Path.Combine(@"D:\Project\ml_cache\models", "vlm");
            psi.Environment["HF_HOME"] = @"D:\Project\ml_cache\huggingface";
            psi.Environment["HUGGINGFACE_HUB_CACHE"] = @"D:\Project\ml_cache\huggingface";
            psi.Environment["GAMESECRETARY_PARENT_PID"] = Process.GetCurrentProcess().Id.ToString();
        }
        catch
        {
        }

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };
        p.OutputDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(outLog, ev.Data); };
        p.ErrorDataReceived += (_, ev) => { if (ev.Data != null) AppendLineSafe(errLog, ev.Data); };

        if (!p.Start())
        {
            throw new InvalidOperationException("Failed to start backend process");
        }

        p.BeginOutputReadLine();
        p.BeginErrorReadLine();
        _proc = p;

        try
        {
            Win32Job.EnsureJob(ref _job);
            Win32Job.Assign(_job, p);
        }
        catch
        {
        }

        try
        {
            var state = new
            {
                pid = p.Id,
                host = ApiHost,
                port = ApiPort,
                py,
                script,
                @out = outLog,
                err = errLog,
                updated_at = DateTime.UtcNow.ToString("o"),
                note = "started_by_gamesecretary_app"
            };
            File.WriteAllText(statePath, JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch
        {
        }

        var ok = await WaitApiReadyAsync(timeoutSec: 60);
        if (!ok)
        {
            throw new TimeoutException($"Backend did not become ready: {ApiBase}/status");
        }
    }

    public async Task RestartAsync(int warmupTimeoutSec)
    {
        Stop();
        await Task.Delay(250);
        await StartAsync();
        try
        {
            var warm = (Environment.GetEnvironmentVariable("GAMESECRETARY_WARMUP_ON_START") ?? "").Trim();
            if (warm == "1" && warmupTimeoutSec > 0)
            {
                await WarmupLocalVlmAsync(timeoutSec: warmupTimeoutSec);
            }
        }
        catch
        {
        }
    }

    private static void TryKillTcpListener(int port)
    {
        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = "netstat",
                Arguments = "-ano -p tcp",
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };

            using var p = Process.Start(psi);
            if (p == null) return;
            var txt = p.StandardOutput.ReadToEnd();
            try { p.WaitForExit(1500); } catch { }

            var re = new Regex(@"\s+TCP\s+[^\s]+:" + port + @"\s+[^\s]+\s+LISTENING\s+(\d+)", RegexOptions.IgnoreCase);
            var ms = re.Matches(txt ?? "");
            if (ms == null || ms.Count <= 0) return;

            foreach (Match m in ms)
            {
                try
                {
                    if (m == null || !m.Success) continue;
                    if (!int.TryParse(m.Groups[1].Value, out var pid)) continue;
                    if (pid <= 0) continue;
                    try
                    {
                        var proc = Process.GetProcessById(pid);
                        proc.Kill(entireProcessTree: true);
                    }
                    catch
                    {
                    }
                }
                catch
                {
                }
            }
        }
        catch
        {
        }
    }

    public async Task<bool> WarmupLocalVlmAsync(int timeoutSec)
    {
        var url = $"{ApiBase}/local_vlm/warmup?max_new_tokens=16";
        using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(timeoutSec) };
        try
        {
            var resp = await http.PostAsync(url, content: null);
            if (!resp.IsSuccessStatusCode)
            {
                return false;
            }

            // Read JSON to surface parse errors early (optional)
            var txt = await resp.Content.ReadAsStringAsync();
            try { JsonDocument.Parse(txt); } catch { }
            return true;
        }
        catch
        {
            return false;
        }
    }

    public void Stop()
    {
        try
        {
            if (_proc is null)
            {
                try { RequestBackendShutdown(); } catch { }
                try { KillBackendByCommandLine(); } catch { }
                try { KillBackendByStatePid(); } catch { }
                try { TryKillTcpListener(ApiPort); } catch { }
                try { Win32Job.Close(ref _job); } catch { }
                return;
            }

            if (!_proc.HasExited)
            {
                try { RequestBackendShutdown(); } catch { }
                try { _proc.Kill(entireProcessTree: true); } catch { }
            }
        }
        finally
        {
            try { _proc?.Dispose(); } catch { }
            _proc = null;
            try { KillBackendByCommandLine(); } catch { }
            try { KillBackendByStatePid(); } catch { }
            try { TryKillTcpListener(ApiPort); } catch { }
            try { Win32Job.Close(ref _job); } catch { }
        }
    }

    private void RequestBackendShutdown()
    {
        try
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            var task = client.PostAsync($"{ApiBase}/shutdown", null);
            task.Wait(1500);
        }
        catch
        {
        }
    }

    private void KillBackendByCommandLine()
    {
        try
        {
            var logPath = Path.Combine(LogsDir, "kill_debug.txt");
            var script = "$ErrorActionPreference='SilentlyContinue';" +
                         $"$log='{logPath.Replace("'", "''")}';" +
                         "function Log($msg) { Add-Content -Path $log -Value ('['+(Get-Date).ToString('HH:mm:ss')+'] ' + $msg) -ErrorAction SilentlyContinue };" +
                         "Log 'Starting cleanup...';" +
                         "$me = [System.Diagnostics.Process]::GetCurrentProcess().Id;" + 
                         "$procs = Get-CimInstance Win32_Process;" +
                         "foreach($p in $procs){" +
                         "  if($p.ProcessId -eq $me){ continue }" + 
                         "  $cl = $p.CommandLine;" +
                         "  $n = $p.Name;" +
                         "  if($n -ne 'python.exe' -and $n -ne 'pythonw.exe'){ continue }" +
                         "  $hit = $false;" +
                         "  if($cl -like '*run_backend.py*'){ $hit = $true; Log 'Match: run_backend' }" +
                         "  if($cl -like '*server.app*' -or $cl -like '*uvicorn*api/v1*'){ $hit = $true; Log 'Match: uvicorn' }" +
                         "  if($hit){ " +
                         "      Log ('Killing PID=' + $p.ProcessId + ' CL=' + $cl);" +
                         "      Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue" +
                         "  }" +
                         "}";

            var psi = new ProcessStartInfo
            {
                FileName = "powershell",
                Arguments = $"-NoProfile -ExecutionPolicy Bypass -Command \"{script}\"",
                CreateNoWindow = true,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };

            using var p = Process.Start(psi);
            try { p?.WaitForExit(3500); } catch { }

            // Nuclear option: WMIC (in case PowerShell is restricted or fails)
            try
            {
                // Kill python processes running from this repo (matching the path in command line)
                // We use a simplified match to avoid complex escaping issues.
                var repoFolder = "ai game secretary"; 
                var wmicArgs = $"process where \"name='python.exe' and CommandLine like '%{repoFolder}%'\" delete";
                var psiWmic = new ProcessStartInfo
                {
                    FileName = "wmic",
                    Arguments = wmicArgs,
                    CreateNoWindow = true,
                    UseShellExecute = false
                };
                using var pWmic = Process.Start(psiWmic);
                pWmic?.WaitForExit(1000);
            }
            catch { }
        }
        catch
        {
        }
    }

    private void KillBackendByStatePid()
    {
        try
        {
            var root = _repoRoot;
            string statePath;
            if (!string.IsNullOrWhiteSpace(root))
            {
                statePath = Path.Combine(root, "logs", "backend.state.json");
            }
            else
            {
                statePath = Path.Combine(LogsDir, "backend.state.json");
            }
            if (!File.Exists(statePath)) return;
            var txt = File.ReadAllText(statePath);
            using var doc = JsonDocument.Parse(txt);
            if (!doc.RootElement.TryGetProperty("pid", out var pidEl)) return;
            var pid = pidEl.GetInt32();
            if (pid <= 0) return;

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "taskkill",
                    Arguments = $"/PID {pid} /T /F",
                    CreateNoWindow = true,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };
                using var p = Process.Start(psi);
                try { p?.WaitForExit(2500); } catch { }
            }
            catch
            {
                try
                {
                    var proc = Process.GetProcessById(pid);
                    proc.Kill(entireProcessTree: true);
                }
                catch
                {
                }
            }
        }
        catch
        {
        }
    }

    private async Task<bool> WaitApiReadyAsync(int timeoutSec)
    {
        var url = $"{ApiBase}/status";
        using var http = new HttpClient();
        var t0 = DateTime.UtcNow;

        while ((DateTime.UtcNow - t0).TotalSeconds < timeoutSec)
        {
            try
            {
                var resp = await http.GetAsync(url);
                if (resp.IsSuccessStatusCode)
                {
                    return true;
                }
            }
            catch
            {
            }

            try
            {
                if (_proc != null && _proc.HasExited)
                {
                    return false;
                }
            }
            catch
            {
            }

            await Task.Delay(350);
        }

        return false;
    }

    private static bool TcpListening(string host, int port, int timeoutMs)
    {
        try
        {
            using var client = new TcpClient();
            var task = client.ConnectAsync(host, port);
            return task.Wait(timeoutMs) && client.Connected;
        }
        catch
        {
            return false;
        }
    }

    private static string? FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (var i = 0; i < 12 && dir is not null; i++)
        {
            var cand = Path.Combine(dir.FullName, "scripts", "run_backend.py");
            if (File.Exists(cand))
            {
                return dir.FullName;
            }

            dir = dir.Parent;
        }

        return null;
    }

    private static void AppendLineSafe(string path, string line)
    {
        try
        {
            if (line != null)
            {
                line = line.Replace("\0", "");
            }
            File.AppendAllText(path, (line ?? "") + Environment.NewLine, Encoding.UTF8);
        }
        catch
        {
        }
    }
}
