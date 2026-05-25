import { useEffect, useMemo, useState } from 'react';
import yaml from 'js-yaml';
import { marked } from 'marked';
import {
  BookOpen,
  Columns3,
  ExternalLink,
  FileText,
  Github,
  Inbox,
  Layers3,
  Link as LinkIcon,
  ListFilter,
  Menu,
  MessageSquareText,
  Search,
  Table2,
  Tags,
} from 'lucide-react';

const BASE_URL = import.meta.env.BASE_URL;
const PRIORITY_ORDER = { P0: 0, P1: 1, P2: 2 };
const SYSTEM_NAV = [
  { id: 'all', label: 'All Papers', icon: BookOpen },
  { id: 'categories', label: 'Categories', icon: Layers3 },
  { id: 'queue', label: 'Reading Queue', icon: Inbox },
  { id: 'notes', label: 'Notes', icon: FileText },
];

marked.setOptions({
  gfm: true,
  breaks: false,
});

function getInitialQuery() {
  const params = new URLSearchParams(window.location.search);
  return {
    paperId: params.get('paper'),
    view: params.get('view') === 'board' ? 'board' : 'table',
  };
}

function updateUrl({ paperId, view }) {
  const params = new URLSearchParams();
  if (paperId) params.set('paper', paperId);
  if (view) params.set('view', view);
  const query = params.toString();
  window.history.replaceState(null, '', `${window.location.pathname}${query ? `?${query}` : ''}`);
}

function normalizePapers(papers) {
  return [...papers].sort((a, b) => {
    const priorityDiff = (PRIORITY_ORDER[a.priority] ?? 9) - (PRIORITY_ORDER[b.priority] ?? 9);
    if (priorityDiff) return priorityDiff;
    return b.year - a.year || a.title.localeCompare(b.title);
  });
}

function shortTitle(title) {
  return title.replace(/^(CODA|Stem|SALS):\s*/, '');
}

function linkLabel(key) {
  const labels = {
    arxiv: 'arXiv',
    code: 'GitHub',
    github: 'GitHub',
    hugging_face: 'HF Papers',
    kernel_library: 'Kernel',
    neurips: 'NeurIPS',
  };
  return labels[key] ?? key.replace(/_/g, ' ');
}

function linkIcon(key) {
  if (key === 'code' || key === 'github' || key === 'kernel_library') return Github;
  if (key === 'arxiv' || key === 'hugging_face' || key === 'neurips') return FileText;
  return ExternalLink;
}

function PriorityChip({ priority }) {
  return <span className={`priority-chip priority-${priority.toLowerCase()}`}>{priority}</span>;
}

function PhaseBadge({ phase }) {
  const label = phase.replace('inference-', 'inference / ');
  return (
    <span className="phase-badge">
      <span className="phase-dot" />
      {label}
    </span>
  );
}

function ExternalLinks({ links }) {
  return (
    <div className="link-list">
      {Object.entries(links ?? {}).map(([key, url]) => {
        const Icon = linkIcon(key);
        return (
          <a className="paper-link" href={url} key={key} rel="noreferrer" target="_blank">
            <Icon size={15} />
            {linkLabel(key)}
          </a>
        );
      })}
    </div>
  );
}

function Toolbar({
  categories,
  category,
  priorities,
  priority,
  search,
  setCategory,
  setPriority,
  setSearch,
  setView,
  view,
}) {
  return (
    <div className="toolbar">
      <label className="search-box">
        <Search size={18} />
        <input
          aria-label="Search papers"
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Search papers..."
          value={search}
        />
        <span className="shortcut">⌘K</span>
      </label>
      <label className="select-field">
        <span>Category</span>
        <select aria-label="Category filter" onChange={(event) => setCategory(event.target.value)} value={category}>
          <option value="all">All</option>
          {categories.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>
      <label className="select-field">
        <span>Priority</span>
        <select aria-label="Priority filter" onChange={(event) => setPriority(event.target.value)} value={priority}>
          <option value="all">All</option>
          {priorities.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </label>
      <div className="view-toggle" aria-label="View mode">
        <button className={view === 'table' ? 'active' : ''} onClick={() => setView('table')} type="button">
          <Table2 size={16} />
          Table
        </button>
        <button className={view === 'board' ? 'active' : ''} onClick={() => setView('board')} type="button">
          <Columns3 size={16} />
          Board
        </button>
      </div>
    </div>
  );
}

function PaperTable({ papers, selectedId, setSelectedId }) {
  return (
    <div className="table-shell">
      <table>
        <thead>
          <tr>
            <th>Priority</th>
            <th>Paper</th>
            <th>System Layer</th>
            <th>Phase</th>
            <th>Tags</th>
            <th>Links</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>
          {papers.map((paper) => (
            <tr
              className={paper.id === selectedId ? 'selected' : ''}
              key={paper.id}
              onClick={() => setSelectedId(paper.id)}
            >
              <td>
                <PriorityChip priority={paper.priority} />
              </td>
              <td>
                <button className="paper-title-button" type="button">
                  <strong>{shortTitle(paper.title)}</strong>
                  <span>{paper.year}</span>
                </button>
              </td>
              <td>{paper.category}</td>
              <td>
                <PhaseBadge phase={paper.phase} />
              </td>
              <td>
                <div className="tag-list">
                  {paper.tags.map((tag) => (
                    <span className="tag" key={tag}>
                      {tag}
                    </span>
                  ))}
                </div>
              </td>
              <td>
                <ExternalLinks links={paper.links} />
              </td>
              <td>
                <button className="note-button" title="Open note" type="button">
                  <MessageSquareText size={18} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Board({ categories, papers, selectedId, setSelectedId }) {
  return (
    <div className="board-grid">
      {categories.map((category) => {
        const categoryPapers = papers.filter((paper) => paper.category === category);
        return (
          <section className="board-column" key={category}>
            <header>
              <h2>{category}</h2>
              <span>{categoryPapers.length}</span>
            </header>
            <div className="board-cards">
              {categoryPapers.map((paper) => (
                <button
                  className={`paper-card ${paper.id === selectedId ? 'selected' : ''}`}
                  key={paper.id}
                  onClick={() => setSelectedId(paper.id)}
                  type="button"
                >
                  <div className="card-topline">
                    <PriorityChip priority={paper.priority} />
                    <PhaseBadge phase={paper.phase} />
                  </div>
                  <strong>{shortTitle(paper.title)}</strong>
                  <span className="card-year">{paper.year}</span>
                  <div className="tag-list">
                    {paper.tags.slice(0, 3).map((tag) => (
                      <span className="tag" key={tag}>
                        {tag}
                      </span>
                    ))}
                  </div>
                  <div className="card-links">
                    <ExternalLinks links={paper.links} />
                    <MessageSquareText size={17} />
                  </div>
                </button>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

function DetailPanel({ markdown, paper, status }) {
  const html = useMemo(() => (markdown ? marked.parse(markdown) : ''), [markdown]);

  return (
    <aside className="detail-panel">
      {paper ? (
        <>
          <div className="detail-header">
            <div>
              <h2>{paper.title}</h2>
              <div className="detail-meta">
                <PriorityChip priority={paper.priority} />
                <span>{paper.year}</span>
                <span>{paper.category}</span>
              </div>
            </div>
            <a className="icon-link" href={`${BASE_URL}content/${paper.note}`} rel="noreferrer" target="_blank" title="Markdown note">
              <FileText size={18} />
            </a>
          </div>
          <div className="detail-links">
            <ExternalLinks links={paper.links} />
            <a className="copy-link" href={`?paper=${paper.id}&view=table`}>
              <LinkIcon size={15} />
              state link
            </a>
          </div>
          <div className="markdown-body">
            {status === 'loading' && <p className="muted">Loading Markdown note...</p>}
            {status === 'error' && <p className="muted">Markdown note failed to load.</p>}
            {status === 'ready' && <div dangerouslySetInnerHTML={{ __html: html }} />}
          </div>
        </>
      ) : (
        <div className="empty-detail">
          <FileText size={24} />
          <p>Select a paper to read its Markdown note.</p>
        </div>
      )}
    </aside>
  );
}

function Sidebar({ activeNav, setActiveNav, setCategory, setPriority, setSearch, setView }) {
  function handleNav(id) {
    setActiveNav(id);
    if (id === 'all') {
      setSearch('');
      setCategory('all');
      setPriority('all');
      setView('table');
    }
    if (id === 'categories') {
      setView('board');
    }
    if (id === 'queue') {
      setPriority('P0');
      setView('table');
    }
  }

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <BookOpen size={18} />
        </div>
        <span>LLM Systems Papers</span>
        <Menu size={18} />
      </div>
      <nav>
        {SYSTEM_NAV.map(({ id, label, icon: Icon }) => (
          <button className={activeNav === id ? 'active' : ''} key={id} onClick={() => handleNav(id)} type="button">
            <Icon size={18} />
            {label}
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <ListFilter size={17} />
        Markdown-first notes
      </div>
    </aside>
  );
}

export default function App() {
  const initialQuery = useMemo(getInitialQuery, []);
  const [papers, setPapers] = useState([]);
  const [selectedId, setSelectedId] = useState(initialQuery.paperId);
  const [view, setView] = useState(initialQuery.view);
  const [category, setCategory] = useState('all');
  const [priority, setPriority] = useState('all');
  const [search, setSearch] = useState('');
  const [markdown, setMarkdown] = useState('');
  const [noteStatus, setNoteStatus] = useState('idle');
  const [loadError, setLoadError] = useState('');
  const [activeNav, setActiveNav] = useState('all');

  useEffect(() => {
    async function loadPapers() {
      try {
        const response = await fetch(`${BASE_URL}content/papers.yml`);
        if (!response.ok) throw new Error(`papers.yml returned ${response.status}`);
        const text = await response.text();
        const parsed = yaml.load(text);
        const normalized = normalizePapers(parsed.papers ?? []);
        setPapers(normalized);
        setSelectedId((current) => current ?? normalized.find((paper) => paper.priority === 'P0')?.id ?? normalized[0]?.id);
      } catch (error) {
        setLoadError(error.message);
      }
    }

    loadPapers();
  }, []);

  const selectedPaper = papers.find((paper) => paper.id === selectedId);

  useEffect(() => {
    if (!selectedPaper) return;
    updateUrl({ paperId: selectedPaper.id, view });
  }, [selectedPaper, view]);

  useEffect(() => {
    async function loadMarkdown() {
      if (!selectedPaper) return;
      setNoteStatus('loading');
      setMarkdown('');
      try {
        const response = await fetch(`${BASE_URL}content/${selectedPaper.note}`);
        if (!response.ok) throw new Error(`${selectedPaper.note} returned ${response.status}`);
        const text = await response.text();
        setMarkdown(text);
        setNoteStatus('ready');
      } catch {
        setNoteStatus('error');
      }
    }

    loadMarkdown();
  }, [selectedPaper]);

  const categories = useMemo(() => [...new Set(papers.map((paper) => paper.category))], [papers]);
  const priorities = useMemo(() => [...new Set(papers.map((paper) => paper.priority))], [papers]);

  const filteredPapers = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return papers.filter((paper) => {
      const matchesCategory = category === 'all' || paper.category === category;
      const matchesPriority = priority === 'all' || paper.priority === priority;
      const searchable = [paper.title, paper.category, paper.phase, paper.priority, paper.year, ...paper.tags]
        .join(' ')
        .toLowerCase();
      return matchesCategory && matchesPriority && (!needle || searchable.includes(needle));
    });
  }, [category, papers, priority, search]);

  useEffect(() => {
    if (!filteredPapers.length || filteredPapers.some((paper) => paper.id === selectedId)) return;
    setSelectedId(filteredPapers[0].id);
  }, [filteredPapers, selectedId]);

  return (
    <div className="app-shell">
      <Sidebar
        activeNav={activeNav}
        setActiveNav={setActiveNav}
        setCategory={setCategory}
        setPriority={setPriority}
        setSearch={setSearch}
        setView={setView}
      />
      <main className="main-panel">
        <header className="page-header">
          <div>
            <h1>Paper Database</h1>
            <p>{papers.length} papers · Markdown notes as source of truth</p>
          </div>
          <a className="repository-link" href="https://github.com/ZhenWZ/llm-systems-papers" rel="noreferrer" target="_blank">
            <Github size={17} />
            GitHub
          </a>
        </header>
        <Toolbar
          categories={categories}
          category={category}
          priorities={priorities}
          priority={priority}
          search={search}
          setCategory={setCategory}
          setPriority={setPriority}
          setSearch={setSearch}
          setView={setView}
          view={view}
        />
        {loadError ? (
          <div className="load-error">Failed to load papers: {loadError}</div>
        ) : (
          <>
            <div className="result-bar">
              <span>{filteredPapers.length} papers</span>
              <span>
                <Tags size={15} />
                {categories.length} system layers
              </span>
            </div>
            {view === 'table' ? (
              <>
                <PaperTable papers={filteredPapers} selectedId={selectedId} setSelectedId={setSelectedId} />
                <section className="board-preview">
                  <Board categories={categories} papers={filteredPapers} selectedId={selectedId} setSelectedId={setSelectedId} />
                </section>
              </>
            ) : (
              <Board categories={categories} papers={filteredPapers} selectedId={selectedId} setSelectedId={setSelectedId} />
            )}
          </>
        )}
      </main>
      <DetailPanel markdown={markdown} paper={selectedPaper} status={noteStatus} />
    </div>
  );
}
