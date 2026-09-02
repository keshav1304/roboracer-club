export type DocSlug = "waveshare" | "lehigh";

export interface DocHeading {
  id: string;
  title: string;
}

export interface DocData {
  content: string;
  toc: DocHeading[];
  sourceUrl: string;
}

const GITHUB_REPO = process.env.DOCS_GITHUB_REPO ?? "keshav1304/roboracer-mini";
const GITHUB_BRANCH = process.env.DOCS_GITHUB_BRANCH ?? "main";

const REVALIDATE_SECONDS = Number(process.env.DOCS_REVALIDATE_SECONDS ?? 60);

const DOC_FOLDERS: Record<DocSlug, string> = {
  waveshare: "donkeycar_documentation",
  lehigh: "lehigh_116_documentation",
};

const SOURCE_URLS: Record<DocSlug, string> = {
  waveshare: `https://github.com/${GITHUB_REPO}/blob/${GITHUB_BRANCH}/donkeycar_documentation/DOCUMENTATION.md`,
  lehigh: `https://github.com/${GITHUB_REPO}/blob/${GITHUB_BRANCH}/lehigh_116_documentation/DOCUMENTATION.md`,
};

function rawGithubUrl(folder: string, filePath: string): string {
  return `https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/${folder}/${filePath}`;
}

export function getGithubAssetBase(slug: DocSlug): string {
  return rawGithubUrl(DOC_FOLDERS[slug], "pictures");
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-");
}

export function headingId(text: string): string {
  const weekMatch = text.match(/^Week (\d+)/);
  if (weekMatch) return `week-${weekMatch[1]}`;
  return slugify(text);
}

function extractHeadings(content: string): DocHeading[] {
  const headings: DocHeading[] = [];
  const regex = /^## (.+)$/gm;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(content)) !== null) {
    const title = match[1].replace(/&amp;/g, "&");
    if (title === "TL;DR" || title === "Table of Contents") continue;
    headings.push({ id: headingId(title), title });
  }
  return headings;
}

export function preprocessDocMarkdown(content: string, assetBase: DocSlug): string {
  const picturesBase = getGithubAssetBase(assetBase);

  let processed = content
    .replace(/!\[([^\]]*)\]\(pictures\//g, `![$1](${picturesBase}/`)
    .replace(/src="pictures\//g, `src="${picturesBase}/`)
    .replace(/\[Teleop guide\]\(\.\/teleop\.md\)/g, "Teleop guide");

  processed = processed.replace(/\n## Table of Contents\n[\s\S]*?(?=\n---\n|\n## )/, "");
  processed = processed.replace(/^# .+\n+/, "");

  processed = processed.replace(
    /## TL;DR\n([\s\S]*?)(?=\n## |\n---\n)/,
    '<div class="doc-callout">\n\n## TL;DR\n$1\n\n</div>\n'
  );

  processed = processed.replace(/\$([^$\n]+)\$/g, "`$1`");

  // Section headings already render a top border; drop markdown thematic breaks.
  processed = processed.replace(/\n(?:---|\*\*\*|___)\n/g, "\n\n");

  return processed;
}

export async function readDoc(slug: DocSlug): Promise<DocData> {
  const url = rawGithubUrl(DOC_FOLDERS[slug], "DOCUMENTATION.md");
  const response = await fetch(
    url,
    process.env.NODE_ENV === "development"
      ? { cache: "no-store" }
      : { next: { revalidate: REVALIDATE_SECONDS } }
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch documentation for "${slug}" (${response.status})`);
  }

  const raw = await response.text();
  const content = preprocessDocMarkdown(raw, slug);

  return {
    content,
    toc: extractHeadings(content),
    sourceUrl: SOURCE_URLS[slug],
  };
}
