import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus as themeDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { vs as themeLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Send,
  Upload,
  FileText,
  Sun,
  Moon,
  Palette,
  Trash2,
  Copy,
  Download,
  Link,
  Bot,
  User,
  Zap,
  Cpu,
  Search,
  Sparkles,
  Menu,
  X
} from "lucide-react";
import { askStreamGen, ingestFromUrl, API_BASE, type SourceItem } from "./api";

type Msg = {
  id: string;
  role: "user" | "assistant";
  text: string;
  imgs?: string[];
  sources?: SourceItem[];
};

function useAutoScroll(dep: any) {
  const scroller = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [dep]);
  return scroller;
}

export default function App() {
  const [q, setQ] = useState("");
  const [pdfUrl, setPdfUrl] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [busy, setBusy] = useState(false);
  const [ingestPct, setIngestPct] = useState(0);
  const [theme, setTheme] = useState<"dark" | "light">(() => (localStorage.getItem("vrag_theme") as any) || "dark");
  const [compact, setCompact] = useState<boolean>(() => localStorage.getItem("vrag_compact") === "1");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const chatRef = useAutoScroll(msgs);

  // Load persisted messages once
  useEffect(() => {
    try {
      const raw = localStorage.getItem("vrag_msgs");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setMsgs(parsed);
      }
    } catch {}
  }, []);

  // Persist messages
  useEffect(() => {
    try { localStorage.setItem("vrag_msgs", JSON.stringify(msgs)); } catch {}
  }, [msgs]);

  // Apply theme/compact classes and persist
  useEffect(() => {
    const cls = document.body.classList;
    if (theme === "light") cls.add("light"); else cls.remove("light");
    if (compact) cls.add("compact"); else cls.remove("compact");
    localStorage.setItem("vrag_theme", theme);
    localStorage.setItem("vrag_compact", compact ? "1" : "0");
  }, [theme, compact]);

  async function onAsk() {
    const prompt = q.trim();
    if (!prompt || busy) return;
    setQ("");
    const id = crypto.randomUUID();
    setMsgs((m) => [...m, { id: crypto.randomUUID(), role: "user", text: prompt }, { id, role: "assistant", text: "" }]);
    setBusy(true);
    try {
      for await (const evt of askStreamGen(prompt)) {
        if (evt.type === "meta") {
          setMsgs((m) => m.map((x) => (x.id === id ? { ...x, imgs: evt.images, sources: evt.sources } : x)));
        } else if (evt.type === "delta") {
          setMsgs((m) => m.map((x) => (x.id === id ? { ...x, text: x.text + evt.text } : x)));
        }
      }
    } catch (e: any) {
      setMsgs((m) => m.map((x) => (x.id === id ? { ...x, text: `Error: ${String(e)}` } : x)));
    } finally {
      setBusy(false);
    }
  }

  async function onIngestUrl() {
    const url = pdfUrl.trim();
    if (!url || busy) return;
    setBusy(true);
    setIngestPct(90); // pseudo-progress for URL fetch
    try {
      await ingestFromUrl(url);
      alert("Ingested! You can ask now.");
    } catch (e: any) {
      alert("Ingest failed: " + String(e));
    } finally {
      setBusy(false);
      setIngestPct(0);
    }
  }

  async function onIngestFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const f = ev.target.files?.[0];
    if (!f || busy) return;
    await uploadFile(f);
    ev.target.value = "";
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onAsk();
    }
  }

  async function uploadFile(file: File) {
    setBusy(true);
    setIngestPct(0);
    try {
      const form = new FormData();
      form.append("pdf_file", file);
      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/ingest`);
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            setIngestPct(Math.round((e.loaded / e.total) * 100));
          }
        };
        xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(xhr.statusText)));
        xhr.onerror = () => reject(new Error("Network error"));
        xhr.send(form);
      });
      alert("Ingested! You can ask now.");
    } catch (e: any) {
      alert("Ingest failed: " + String(e));
    } finally {
      setBusy(false);
      setIngestPct(0);
    }
  }

  return (
    <div className={`layout ${sidebarOpen ? 'sidebar-open' : ''}`}>
      {/* Mobile sidebar toggle */}
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
      >
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Mobile sidebar overlay */}
      <div
        className="sidebar-overlay"
        onClick={() => setSidebarOpen(false)}
      />

      <aside className="sidebar">
        <div className="brand">
          <Sparkles size={24} />
          <span>Vision‑RAG</span>
        </div>
        <div className="sep" />
        <div className="ingest">
          <div className="ingest-header">
            <FileText size={16} />
            <span className="muted">Document Upload</span>
          </div>

          <div className="url-input-group">
            <input
              className="input"
              placeholder="Enter PDF URL..."
              value={pdfUrl}
              onChange={(e) => setPdfUrl(e.target.value)}
            />
            <button className="button primary" onClick={onIngestUrl} disabled={busy}>
              <Link size={16} />
              <span>Ingest</span>
            </button>
          </div>

          <div
            className="dropzone"
            onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; }}
            onDrop={async (e) => {
              e.preventDefault();
              const file = e.dataTransfer.files?.[0];
              if (file) await uploadFile(file);
            }}
          >
            <Upload size={24} />
            <div>Drag & drop PDF files here</div>
            <div className="muted">or click to browse</div>
          </div>

          {busy && (
            <div className="upload-progress">
              <div className="progress">
                <span style={{ width: `${Math.max(ingestPct, 10)}%` }} />
              </div>
              <span className="muted">Processing... {ingestPct}%</span>
            </div>
          )}

          <div className="file-input-wrapper">
            <input
              id="file-input"
              className="file-input"
              type="file"
              accept="application/pdf"
              onChange={onIngestFile}
            />
            <label htmlFor="file-input" className="file-input-label">
              <Upload size={16} />
              <span>Choose PDF file</span>
            </label>
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="header">
          <div className="title">
            <Bot size={20} />
            <span>Chat</span>
          </div>
          <div className="muted" style={{ marginLeft: 8, display: 'flex', alignItems: 'center', gap: 4 }}>
            <Cpu size={14} />
            <span>ColPali → Gemma‑3</span>
          </div>
          <div style={{ flex: 1 }} />
          <div className="header-controls">
            <button className="button" onClick={() => setTheme(t => t === "dark" ? "light" : "dark")} title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
              {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button className="button" onClick={() => setCompact(c => !c)} title={compact ? "Switch to comfortable view" : "Switch to compact view"}>
              <Palette size={16} />
            </button>
            {/* Heatmap toggle removed since context images are hidden */}
            {/* <button className="button" onClick={() => setShowHeatmaps(h => !h)} title={showHeatmaps ? "Hide heatmaps" : "Show heatmaps"}>
              {showHeatmaps ? <EyeOff size={16} /> : <Eye size={16} />}
            </button> */}
            <button className="button" onClick={() => setMsgs([])} title="Clear conversation">
              <Trash2 size={16} />
            </button>
          </div>
          <div className="keyboard-hints">
            <span className="kbd">Enter</span>
            <span className="muted">to send</span>
            <span className="kbd">Shift</span>+
            <span className="kbd">Enter</span>
            <span className="muted">for newline</span>
          </div>
        </div>

        <div className="chat" ref={chatRef}>
          {msgs.map((m) => (
            <div key={m.id} className={`row ${m.role}`}>
              <div className="avatar">
                {m.role === "user" ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div className="bubble">
                {m.role === "assistant" ? (
                  <button className="copy-btn" title="Copy answer" onClick={() => navigator.clipboard.writeText(m.text)}>
                    <Copy size={12} />
                  </button>
                ) : null}
                {/* Context pages/images are now hidden */}
                {/*
                {m.sources?.length ? (
                  <div className="images-meta">Context pages</div>
                ) : m.imgs?.length ? (
                  <div className="images-meta">Context pages</div>
                ) : null}
                {m.sources?.length ? (
                  <div className="thumbs">
                    {m.sources.map((s, i) => (
                      <div key={i} className="thumb">
                        <img className="page" src={s.image_url} alt="page" />
                        {showHeatmaps && s.heatmap_url ? (
                          <img className="heatmap" src={s.heatmap_url} alt="heatmap" />
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : m.imgs?.length ? (
                  <div className="thumbs">
                    {m.imgs.map((u, i) => (
                      <img key={i} src={u} alt="page" />
                    ))}
                  </div>
                ) : null}
                */}
                <div>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code({inline, className, children, ...props}){
                        const match = /language-(\w+)/.exec(className || "");
                        if (!inline) {
                          return (
                            <SyntaxHighlighter
                              style={theme === "dark" ? themeDark : themeLight}
                              language={match?.[1] || "text"}
                              PreTag="div"
                              customStyle={{ margin: 0 }}
                              {...props}
                            >
                              {String(children).replace(/\n$/, "")}
                            </SyntaxHighlighter>
                          );
                        }
                        return <code className={className} {...props}>{children}</code>;
                      }
                    }}
                  >
                    {m.text || (m.role === "assistant" && busy ? "…" : "")}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          {!msgs.length ? (
            <div className="muted">Ask about your PDFs after ingesting one.</div>
          ) : null}
        </div>

        <div className="composer">
          <div className="composer-inner">
            <textarea
              placeholder="Ask about your documents..."
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={onKeyDown}
            />
            <button className="button primary" onClick={onAsk} disabled={busy || !q.trim()}>
              <Send size={16} />
              <span>Send</span>
            </button>
          </div>
          <div className="keyboard-hints">
            <span className="kbd">Enter</span>
            <span className="muted">to send</span>
            <span className="kbd">Shift</span>+
            <span className="kbd">Enter</span>
            <span className="muted">for newline</span>
          </div>
        </div>
      </main>
    </div>
  );
}
