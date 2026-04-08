using System;
using System.IO;
using System.Threading;
using System.Windows;
using MessageBox = System.Windows.MessageBox;

namespace GameSecretaryApp;

public partial class App : System.Windows.Application
{
    private Mutex? _mutex;

    private static string GetLogPath()
    {
        try
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            for (var i = 0; i < 10 && dir is not null; i++)
            {
                var cand = Path.Combine(dir.FullName, "logs");
                var main = Path.Combine(dir.FullName, "main.py");
                if (Directory.Exists(cand) && File.Exists(main))
                {
                    return Path.Combine(cand, "gamesecretary_app.log");
                }
                dir = dir.Parent;
            }
        }
        catch
        {
        }

        var local = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GameSecretaryApp",
            "gamesecretary_app.log"
        );
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(local) ?? Path.GetTempPath());
        }
        catch
        {
        }
        return local;
    }

    private static void Log(string msg)
    {
        try
        {
            File.AppendAllText(GetLogPath(), $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {msg}{Environment.NewLine}");
        }
        catch
        {
        }
    }

    protected override void OnStartup(StartupEventArgs e)
    {
        Log("App starting...");

        DispatcherUnhandledException += (_, ev) =>
        {
            Log("DispatcherUnhandledException: " + ev.Exception);
            try { MessageBox.Show(ev.Exception.Message, "GameSecretaryApp crash", MessageBoxButton.OK, MessageBoxImage.Error); } catch { }
            ev.Handled = true;
        };

        AppDomain.CurrentDomain.UnhandledException += (_, ev) =>
        {
            Log("UnhandledException: " + (ev.ExceptionObject?.ToString() ?? "(null)"));
        };

        bool createdNew;
        _mutex = new Mutex(true, "GameSecretaryApp.SingleInstance", out createdNew);
        if (!createdNew)
        {
            Log("Another instance detected -> shutting down");
            MessageBox.Show("GameSecretaryApp is already running.", "GameSecretaryApp", MessageBoxButton.OK, MessageBoxImage.Information);
            Shutdown();
            return;
        }

        base.OnStartup(e);
        Log("App started");
    }

    protected override void OnExit(ExitEventArgs e)
    {
        Log("App exiting...");
        try
        {
            BackendManager.Instance.Stop();
        }
        catch
        {
        }

        try
        {
            _mutex?.ReleaseMutex();
            _mutex?.Dispose();
        }
        catch
        {
        }

        base.OnExit(e);
    }
}
