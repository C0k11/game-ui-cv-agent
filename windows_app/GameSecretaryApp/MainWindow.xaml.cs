using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using Microsoft.Web.WebView2.Core;

namespace GameSecretaryApp;

public partial class MainWindow : Window
{
    private const int HotkeyIdToggleOverlay = 0xA115;
    private OverlayWindow? _overlay;
    private IntPtr _hwnd;
    private HwndSource? _source;
    private bool _isRestarting;

    public MainWindow()
    {
        InitializeComponent();
        SourceInitialized += OnSourceInitialized;
        Loaded += OnLoaded;
        Closing += OnClosing;
    }

    private void OnSourceInitialized(object? sender, EventArgs e)
    {
        try
        {
            _hwnd = new WindowInteropHelper(this).Handle;
            _source = HwndSource.FromHwnd(_hwnd);
            _source?.AddHook(WndProc);

            var vk = (uint)KeyInterop.VirtualKeyFromKey(Key.O);
            Win32Window.TryRegisterHotKey(
                _hwnd,
                HotkeyIdToggleOverlay,
                Win32Window.MOD_CONTROL | Win32Window.MOD_SHIFT,
                vk
            );
        }
        catch
        {
        }
    }

    private IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
    {
        if (msg == Win32Window.WM_HOTKEY)
        {
            try
            {
                var id = wParam.ToInt32();
                if (id == HotkeyIdToggleOverlay)
                {
                    ToggleOverlay();
                    handled = true;
                }
            }
            catch
            {
            }
        }
        return IntPtr.Zero;
    }

    private void ToggleOverlay()
    {
        try
        {
            if (_overlay == null)
            {
                _overlay = new OverlayWindow
                {
                    TargetWindowTitleSubstring = "Blue Archive",
                };
                _overlay.Hide();
            }

            if (_overlay.IsVisible)
            {
                _overlay.Hide();
            }
            else
            {
                _overlay.Show();
            }
        }
        catch
        {
        }
    }

    private async void OnLoaded(object sender, RoutedEventArgs e)
    {
        SetStatus("Starting backend...");

        var userData = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GameSecretaryApp",
            "WebView2"
        );
        Directory.CreateDirectory(userData);
        var env = await CoreWebView2Environment.CreateAsync(null, userData);
        await WebView.EnsureCoreWebView2Async(env);

        try
        {
            await BackendManager.Instance.StartAsync();
        }
        catch (Exception ex)
        {
            SetStatus("Backend failed");
            MessageBox.Show(ex.Message, "Backend start failed", MessageBoxButton.OK, MessageBoxImage.Error);
            return;
        }

        SetStatus("Ready");
        WebView.CoreWebView2.Navigate(BackendManager.Instance.DashboardUrl);

        try
        {
            var warm = (Environment.GetEnvironmentVariable("GAMESECRETARY_WARMUP_ON_START") ?? "").Trim();
            if (warm == "1")
            {
                SetStatus("Warming up model...");
                try { await BackendManager.Instance.WarmupLocalVlmAsync(timeoutSec: 1800); } catch { }
                SetStatus("Ready");
            }
        }
        catch
        {
        }
    }

    private void SetStatus(string msg)
    {
        try { Title = "AI Game Secretary" + (string.IsNullOrWhiteSpace(msg) ? "" : $" ({msg})"); } catch { }
        try { if (TxtStatus != null) TxtStatus.Text = "Status: " + (msg ?? ""); } catch { }
    }

    private void SetUiEnabled(bool enabled)
    {
        try { if (BtnRestart != null) BtnRestart.IsEnabled = enabled; } catch { }
        try { if (BtnReload != null) BtnReload.IsEnabled = enabled; } catch { }
        try { if (BtnToggleOverlay != null) BtnToggleOverlay.IsEnabled = enabled; } catch { }
        try { if (BtnOpenLogs != null) BtnOpenLogs.IsEnabled = enabled; } catch { }
        try { if (BtnOpenBrowser != null) BtnOpenBrowser.IsEnabled = enabled; } catch { }
    }

    private async void BtnRestart_Click(object sender, RoutedEventArgs e)
    {
        if (_isRestarting) return;
        _isRestarting = true;
        SetUiEnabled(false);
        SetStatus("Restarting backend...");
        try
        {
            await BackendManager.Instance.RestartAsync(warmupTimeoutSec: 1800);
            SetStatus("Ready");
            try
            {
                WebView.CoreWebView2?.Navigate(BackendManager.Instance.DashboardUrl);
            }
            catch
            {
            }
        }
        catch (Exception ex)
        {
            SetStatus("Restart failed");
            try { MessageBox.Show(ex.Message, "Restart failed", MessageBoxButton.OK, MessageBoxImage.Error); } catch { }
        }
        finally
        {
            _isRestarting = false;
            SetUiEnabled(true);
        }
    }

    private void BtnReload_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            WebView.CoreWebView2?.Reload();
        }
        catch
        {
            try { WebView.Source = new Uri(BackendManager.Instance.DashboardUrl); } catch { }
        }
    }

    private void BtnToggleOverlay_Click(object sender, RoutedEventArgs e)
    {
        ToggleOverlay();
    }

    private void BtnOpenLogs_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var dir = BackendManager.Instance.LogsDir;
            try { Directory.CreateDirectory(dir); } catch { }
            Process.Start(new ProcessStartInfo
            {
                FileName = dir,
                UseShellExecute = true,
            });
        }
        catch
        {
        }
    }

    private void BtnOpenBrowser_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = BackendManager.Instance.DashboardUrl,
                UseShellExecute = true,
            });
        }
        catch
        {
        }
    }

    private void OnClosing(object? sender, System.ComponentModel.CancelEventArgs e)
    {
        try
        {
            BackendManager.Instance.Stop();
        }
        catch
        {
        }

        try { Win32Window.TryUnregisterHotKey(_hwnd, HotkeyIdToggleOverlay); } catch { }
        try { _source?.RemoveHook(WndProc); } catch { }
        try { _overlay?.Close(); } catch { }
    }
}
