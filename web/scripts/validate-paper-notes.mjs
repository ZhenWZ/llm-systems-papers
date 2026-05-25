import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(scriptDir, "..");
const repoRoot = path.resolve(webRoot, "..");

const papersYamlPath = path.join(repoRoot, "data", "papers.yml");
const readmePath = path.join(repoRoot, "README.md");
const categoriesDir = path.join(repoRoot, "categories");

const requiredPaperFields = [
  "id",
  "title",
  "year",
  "priority",
  "category",
  "phase",
  "links",
  "note",
  "tags",
];

const requiredSections = [
  "Metadata",
  "一句话结论",
  "Related Works",
  "问题与背景",
  "方法与系统设计",
  "创新点",
  "实验与结论",
  "局限性与风险",
  "对 LLM Systems 的启发",
  "复现/阅读建议",
];

const requiredMetadataLabels = [
  "Year",
  "Authors",
  "Category",
  "Priority",
  "Links",
  "Keywords",
];

const errors = [];

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function addError(message) {
  errors.push(message);
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function headingIndex(markdown, level, heading) {
  const marker = "#".repeat(level);
  const pattern = new RegExp(`^${marker}\\s+${escapeRegExp(heading)}\\s*$`);
  return markdown.split(/\r?\n/).findIndex((line) => pattern.test(line));
}

function hasHeading(markdown, level, heading) {
  return headingIndex(markdown, level, heading) !== -1;
}

function hasMetadataLabel(markdown, label) {
  return new RegExp(`^-\\s+${escapeRegExp(label)}:\\s+.+$`, "m").test(markdown);
}

function includesNoteLink(markdown, notePath) {
  return markdown.includes(notePath) || markdown.includes(`../${notePath}`);
}

function isSafeRelativeMarkdownPath(notePath) {
  return (
    typeof notePath === "string" &&
    notePath.startsWith("papers/") &&
    notePath.endsWith(".md") &&
    !path.isAbsolute(notePath) &&
    !notePath.split("/").includes("..")
  );
}

if (!fs.existsSync(papersYamlPath)) {
  addError("Missing data/papers.yml.");
}

const papersDoc = fs.existsSync(papersYamlPath)
  ? yaml.load(readText(papersYamlPath))
  : null;

const papers = Array.isArray(papersDoc?.papers) ? papersDoc.papers : [];
if (!Array.isArray(papersDoc?.papers)) {
  addError("data/papers.yml must contain a top-level papers array.");
}

const readme = fs.existsSync(readmePath) ? readText(readmePath) : "";
if (!readme) {
  addError("Missing README.md.");
}

const categoryDocs = fs.existsSync(categoriesDir)
  ? fs
      .readdirSync(categoriesDir)
      .filter((fileName) => fileName.endsWith(".md"))
      .map((fileName) => {
        const filePath = path.join(categoriesDir, fileName);
        return { fileName, content: readText(filePath) };
      })
  : [];

if (!categoryDocs.length) {
  addError("Missing categories/*.md category indexes.");
}

const seenIds = new Set();
const seenNotes = new Set();

for (const [index, paper] of papers.entries()) {
  const label = paper?.id || `papers[${index}]`;

  for (const field of requiredPaperFields) {
    if (paper?.[field] === undefined || paper?.[field] === null || paper?.[field] === "") {
      addError(`${label}: missing required data field "${field}".`);
    }
  }

  if (typeof paper?.id === "string") {
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(paper.id)) {
      addError(`${label}: id must be stable lower-kebab-case.`);
    }
    if (seenIds.has(paper.id)) {
      addError(`${label}: duplicate id "${paper.id}".`);
    }
    seenIds.add(paper.id);
  }

  if (!Number.isInteger(paper?.year)) {
    addError(`${label}: year must be an integer.`);
  }

  if (!["P0", "P1", "P2"].includes(paper?.priority)) {
    addError(`${label}: priority must be P0, P1, or P2.`);
  }

  if (!paper?.links || typeof paper.links !== "object" || Array.isArray(paper.links)) {
    addError(`${label}: links must be a mapping of link labels to URLs.`);
  } else if (!Object.keys(paper.links).length) {
    addError(`${label}: links must contain at least one URL.`);
  }

  if (!Array.isArray(paper?.tags) || !paper.tags.length) {
    addError(`${label}: tags must be a non-empty array.`);
  }

  if (!isSafeRelativeMarkdownPath(paper?.note)) {
    addError(`${label}: note must be a safe papers/*.md path.`);
    continue;
  }

  if (seenNotes.has(paper.note)) {
    addError(`${label}: duplicate note path "${paper.note}".`);
  }
  seenNotes.add(paper.note);

  const notePath = path.resolve(repoRoot, paper.note);
  if (!notePath.startsWith(`${repoRoot}${path.sep}`)) {
    addError(`${label}: note path escapes repository root.`);
    continue;
  }

  if (!fs.existsSync(notePath)) {
    addError(`${label}: note file does not exist at ${paper.note}.`);
    continue;
  }

  const note = readText(notePath);
  const titleMatch = note.trimStart().match(/^#\s+(.+?)\s*(?:\r?\n|$)/);
  if (!titleMatch) {
    addError(`${label}: note must start with a level-1 title.`);
  } else if (titleMatch[1] !== paper.title) {
    addError(`${label}: note title must match data title "${paper.title}".`);
  }

  let previousSectionIndex = -1;
  for (const section of requiredSections) {
    const sectionIndex = headingIndex(note, 2, section);
    if (sectionIndex === -1) {
      addError(`${label}: note is missing "## ${section}".`);
    } else if (sectionIndex <= previousSectionIndex) {
      addError(`${label}: note section "## ${section}" is out of order.`);
    } else {
      previousSectionIndex = sectionIndex;
    }
  }

  for (const metadataLabel of requiredMetadataLabels) {
    if (!hasMetadataLabel(note, metadataLabel)) {
      addError(`${label}: Metadata is missing "- ${metadataLabel}: ...".`);
    }
  }

  if (!new RegExp(`^-\\s+Year:\\s+${paper.year}\\s*$`, "m").test(note)) {
    addError(`${label}: Metadata Year must match data/papers.yml.`);
  }

  if (!new RegExp(`^-\\s+Category:\\s+${escapeRegExp(String(paper.category ?? ""))}\\s*$`, "m").test(note)) {
    addError(`${label}: Metadata Category must match data/papers.yml.`);
  }

  if (!new RegExp(`^-\\s+Priority:\\s+${paper.priority}\\s*$`, "m").test(note)) {
    addError(`${label}: Metadata Priority must match data/papers.yml.`);
  }

  if (!readme.includes(paper.title)) {
    addError(`${label}: README.md must include the paper title.`);
  }

  if (!includesNoteLink(readme, paper.note)) {
    addError(`${label}: README.md must link to ${paper.note}.`);
  }

  const matchingCategoryDocs =
    typeof paper.category === "string"
      ? categoryDocs.filter((doc) => hasHeading(doc.content, 1, paper.category))
      : [];

  if (!matchingCategoryDocs.length) {
    addError(`${label}: no categories/*.md file has "# ${paper.category}".`);
  } else if (!matchingCategoryDocs.some((doc) => includesNoteLink(doc.content, paper.note))) {
    addError(`${label}: category index for "${paper.category}" must link to ${paper.note}.`);
  }
}

if (errors.length) {
  console.error(`Paper note validation failed with ${errors.length} issue(s):`);
  for (const error of errors) {
    console.error(`- ${error}`);
  }
  process.exit(1);
}

console.log(`Paper note validation passed for ${papers.length} paper(s).`);
