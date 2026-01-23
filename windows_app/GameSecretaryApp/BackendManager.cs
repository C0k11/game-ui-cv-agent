using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.RegularExpressions;
using System.Text.Json;
using System.Text;
using System.Threading.Tasks;

namespace GameSecretaryApp;

public sealed class BackendManager
{
    public static BackendManager Instance { get; } = new BackendManager();

    private Process? _proc;
    private string? _repoRoot;

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
            psi.Environment["LOCAL_VLM_DEVICE"] = "cuda";
            psi.Environment["LOCAL_VLM_MODELS_DIR"] = Path.Combine(@"D:\Project\ml_cache\models", "vlm");
            psi.Environment["HF_HOME"] = @"D:\Project\ml_cache\huggingface";
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
            await WarmupLocalVlmAsync(timeoutSec: warmupTimeoutSec);
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

            var re = new Regex(@"\\s+TCP\\s+[^\\s]+:" + port + @"\\s+[^\\s]+\\s+LISTENING\\s+(\\d+)", RegexOptions.IgnoreCase);
            var m = re.Match(txt ?? "");
            if (!m.Success) return;
            if (!int.TryParse(m.Groups[1].Value, out var pid)) return;
            if (pid <= 0) return;

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
                return;
            }

            if (!_proc.HasExited)
            {
                try { _proc.Kill(entireProcessTree: true); } catch { }
            }
        }
        finally
        {
            try { _proc?.Dispose(); } catch { }
            _proc = null;
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
