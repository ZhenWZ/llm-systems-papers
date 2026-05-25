import { cp, mkdir, rm } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const webRoot = resolve(__dirname, '..');
const repoRoot = resolve(webRoot, '..');
const contentRoot = resolve(webRoot, 'public', 'content');

await rm(contentRoot, { recursive: true, force: true });
await mkdir(contentRoot, { recursive: true });
await cp(resolve(repoRoot, 'data', 'papers.yml'), resolve(contentRoot, 'papers.yml'));
await cp(resolve(repoRoot, 'papers'), resolve(contentRoot, 'papers'), {
  recursive: true,
});

console.log('Synced paper metadata and Markdown notes into web/public/content');
