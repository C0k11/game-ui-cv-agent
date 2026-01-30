using System;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;
using System.Windows.Threading;

namespace GameSecretaryApp;

public partial class OverlayWindow : Window
{
    private readonly DispatcherTimer _timer;
    private readonly HttpClient _http;

    public string TargetWindowTitleSubstring { get; set; } = "Blue Archive";

    public OverlayWindow()
    {
        InitializeComponent();
        _http = new HttpClient { Timeout = TimeSpan.FromSeconds(1.2) };
        _timer = new DispatcherTimer(DispatcherPriority.Background)
        {
            Interval = TimeSpan.FromMilliseconds(200)
        };
        _timer.Tick += async (_, _) => await TickAsync();

        Loaded += (_, _) =>
        {
            try { Win32Window.MakeClickThrough(this); } catch { }
            _timer.Start();
        };

        Closed += (_, _) =>
        {
            try { _timer.Stop(); } catch { }
            try { _http.Dispose(); } catch { }
        };
    }

    private async Task TickAsync()
    {
        var scaleX = 1.0;
        var scaleY = 1.0;
        try
        {
            var src = PresentationSource.FromVisual(this);
            if (src?.CompositionTarget != null)
            {
                var m = src.CompositionTarget.TransformToDevice;
                scaleX = m.M11;
                scaleY = m.M22;
            }
        }
        catch { }

        try
        {
            if (Win32Window.TryGetClientRectOnScreen(TargetWindowTitleSubstring, out var rect))
            {
                if (scaleX > 0 && scaleY > 0)
                {
                    Left = rect.Left / scaleX;
                    Top = rect.Top / scaleY;
                    Width = Math.Max(1, rect.Width / scaleX);
                    Height = Math.Max(1, rect.Height / scaleY);
                }
                else
                {
                    Left = rect.Left;
                    Top = rect.Top;
                    Width = Math.Max(1, rect.Width);
                    Height = Math.Max(1, rect.Height);
                }
                if (!IsVisible) Show();
            }
            else
            {
                Hide();
                return;
            }
        }
        catch
        {
        }

        JsonDocument? doc = null;
        try
        {
            var url = $"{BackendManager.Instance.ApiBase}/status";
            var json = await _http.GetStringAsync(url);
            doc = JsonDocument.Parse(json);
        }
        catch
        {
        }

        try
        {
            if (doc != null && doc.RootElement.ValueKind == JsonValueKind.Object)
            {
                if (doc.RootElement.TryGetProperty("agent_cfg", out var cfg) && cfg.ValueKind == JsonValueKind.Object)
                {
                    var wt = TryGetString(cfg, "window_title");
                    if (!string.IsNullOrWhiteSpace(wt))
                    {
                        TargetWindowTitleSubstring = wt;
                    }
                }
            }
        }
        catch
        {
        }

        try
        {
            RenderAction(doc, scaleX, scaleY);
        }
        catch
        {
            try { CanvasRoot.Children.Clear(); } catch { }
        }
        finally
        {
            try { doc?.Dispose(); } catch { }
        }
    }

    private static bool TryGetBool(JsonElement el, string name)
    {
        if (el.ValueKind != JsonValueKind.Object) return false;
        if (!el.TryGetProperty(name, out var v)) return false;
        if (v.ValueKind == JsonValueKind.True) return true;
        if (v.ValueKind == JsonValueKind.False) return false;
        if (v.ValueKind == JsonValueKind.String && bool.TryParse(v.GetString(), out var b)) return b;
        return false;
    }

    private static string TryGetString(JsonElement el, string name)
    {
        if (el.ValueKind != JsonValueKind.Object) return "";
        if (!el.TryGetProperty(name, out var v)) return "";
        if (v.ValueKind == JsonValueKind.String) return v.GetString() ?? "";
        return v.ToString();
    }

    private static bool TryGetPoint(JsonElement action, string key, out Point p)
    {
        p = new Point();
        if (!action.TryGetProperty(key, out var arr) || arr.ValueKind != JsonValueKind.Array) return false;
        if (arr.GetArrayLength() != 2) return false;
        try
        {
            var x = arr[0].GetInt32();
            var y = arr[1].GetInt32();
            p = new Point(x, y);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryGetRect(JsonElement action, string key, out Rect r)
    {
        r = new Rect();
        if (!action.TryGetProperty(key, out var arr) || arr.ValueKind != JsonValueKind.Array) return false;
        if (arr.GetArrayLength() != 4) return false;
        try
        {
            var x1 = arr[0].GetInt32();
            var y1 = arr[1].GetInt32();
            var x2 = arr[2].GetInt32();
            var y2 = arr[3].GetInt32();
            r = new Rect(x1, y1, Math.Max(1, x2 - x1), Math.Max(1, y2 - y1));
            return true;
        }
        catch
        {
            return false;
        }
    }

    private void RenderAction(JsonDocument? doc, double scaleX, double scaleY)
    {
        CanvasRoot.Children.Clear();
        if (doc is null) return;

        if (scaleX <= 0) scaleX = 1.0;
        if (scaleY <= 0) scaleY = 1.0;

        if (!doc.RootElement.TryGetProperty("last_action", out var act) || act.ValueKind != JsonValueKind.Object)
        {
            return;
        }

        var a = TryGetString(act, "action").ToLowerInvariant().Trim();
        var reason = TryGetString(act, "reason");
        var blocked = TryGetBool(act, "_blocked");
        var exploration = TryGetBool(act, "_exploration");

        var color = blocked
            ? Color.FromArgb(220, 220, 40, 40)
            : (exploration ? Color.FromArgb(220, 240, 180, 30) : Color.FromArgb(220, 50, 220, 120));
        var stroke = new SolidColorBrush(color);

        try
        {
            JsonElement p;
            if (act.TryGetProperty("_perception_client", out p) && p.ValueKind == JsonValueKind.Object)
            {
            }
            else if (act.TryGetProperty("_perception", out p) && p.ValueKind == JsonValueKind.Object)
            {
            }
            else
            {
                p = default;
            }

            if (p.ValueKind == JsonValueKind.Object && p.TryGetProperty("items", out var items) && items.ValueKind == JsonValueKind.Array)
            {
                var bboxBrush = new SolidColorBrush(Color.FromArgb(160, 90, 160, 255));
                var n = 0;
                foreach (var it in items.EnumerateArray())
                {
                    if (n >= 25) break;
                    if (it.ValueKind != JsonValueKind.Object) continue;
                    if (!TryGetRect(it, "bbox", out var r)) continue;
                    DrawRect(r, bboxBrush, scaleX, scaleY);
                    n++;
                }
            }
        }
        catch
        {
        }

        if (a == "click")
        {
            if (TryGetPoint(act, "target_client", out var p) || TryGetPoint(act, "target", out p))
            {
                DrawCross(p, stroke, scaleX, scaleY);
            }
            else if (TryGetRect(act, "bbox_client", out var r) || TryGetRect(act, "bbox", out r))
            {
                DrawRect(r, stroke, scaleX, scaleY);
                DrawCross(new Point(r.X + r.Width / 2.0, r.Y + r.Height / 2.0), stroke, scaleX, scaleY);
            }
        }
        else if (a == "swipe")
        {
            if ((TryGetPoint(act, "from_client", out var p1) || TryGetPoint(act, "from", out p1))
                && (TryGetPoint(act, "to_client", out var p2) || TryGetPoint(act, "to", out p2)))
            {
                DrawLine(p1, p2, stroke, scaleX, scaleY);
                DrawCross(p1, stroke, scaleX, scaleY);
                DrawCross(p2, stroke, scaleX, scaleY);
            }
        }
        else if (a == "wait")
        {
        }

        var text = $"action={a} blocked={blocked} exploration={exploration}";
        if (!string.IsNullOrWhiteSpace(reason))
        {
            var r = reason.Replace("\r", " ").Replace("\n", " ");
            if (r.Length > 160) r = r.Substring(0, 160);
            text += "\n" + r;
        }

        DrawLabel(text, stroke);
    }

    private void DrawLine(Point p1, Point p2, Brush stroke, double sx, double sy)
    {
        var line = new Line
        {
            X1 = p1.X / sx,
            Y1 = p1.Y / sy,
            X2 = p2.X / sx,
            Y2 = p2.Y / sy,
            Stroke = stroke,
            StrokeThickness = 3,
        };
        CanvasRoot.Children.Add(line);
    }

    private void DrawRect(Rect r, Brush stroke, double sx, double sy)
    {
        var rect = new Rectangle
        {
            Width = r.Width / sx,
            Height = r.Height / sy,
            Stroke = stroke,
            StrokeThickness = 2,
        };
        Canvas.SetLeft(rect, r.X / sx);
        Canvas.SetTop(rect, r.Y / sy);
        CanvasRoot.Children.Add(rect);
    }

    private void DrawCross(Point p, Brush stroke, double sx, double sy)
    {
        var px = p.X / sx;
        var py = p.Y / sy;
        var s = 10.0;
        var l1 = new Line { X1 = px - s, Y1 = py, X2 = px + s, Y2 = py, Stroke = stroke, StrokeThickness = 3 };
        var l2 = new Line { X1 = px, Y1 = py - s, X2 = px, Y2 = py + s, Stroke = stroke, StrokeThickness = 3 };
        CanvasRoot.Children.Add(l1);
        CanvasRoot.Children.Add(l2);

        var ring = new Ellipse { Width = 26, Height = 26, Stroke = stroke, StrokeThickness = 2 };
        Canvas.SetLeft(ring, px - ring.Width / 2.0);
        Canvas.SetTop(ring, py - ring.Height / 2.0);
        CanvasRoot.Children.Add(ring);
    }

    private void DrawLabel(string msg, Brush stroke)
    {
        var bg = new SolidColorBrush(Color.FromArgb(120, 0, 0, 0));
        var tb = new System.Windows.Controls.TextBlock
        {
            Text = msg,
            Foreground = stroke,
            Background = bg,
            FontSize = 14,
            Padding = new Thickness(8, 6, 8, 6),
        };
        Canvas.SetLeft(tb, 10);
        Canvas.SetTop(tb, 10);
        CanvasRoot.Children.Add(tb);
    }
}
