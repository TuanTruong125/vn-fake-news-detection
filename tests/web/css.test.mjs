// Test: Validate CSS stylesheets exist and include responsive design patterns
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const stylePath = path.join(repoRoot, 'src', 'web', 'css', 'style.css');
const responsivePath = path.join(repoRoot, 'src', 'web', 'css', 'responsive.css');


test('src/web stylesheets exist and include responsive layout rules', async () => {
  const styleCss = await readFile(stylePath, 'utf8');
  const responsiveCss = await readFile(responsivePath, 'utf8');

  assert.ok(styleCss.includes('.page-shell'), 'style.css should contain page-shell styles');
  assert.ok(styleCss.includes('.topbar'), 'style.css should contain topbar styles');
  assert.ok(styleCss.includes('.result-section'), 'style.css should contain result section styles');
  assert.ok(styleCss.includes('@media (max-width: 767px)'), 'style.css should contain mobile rules');

  assert.ok(responsiveCss.includes('@media (max-width: 1199px)'), 'responsive.css should contain tablet rules');
  assert.ok(responsiveCss.includes('@media (max-width: 767px)'), 'responsive.css should contain mobile rules');
  assert.ok(responsiveCss.includes('.result-section'), 'responsive.css should reference result-section');

  assert.ok(existsSync(stylePath), 'style.css should exist');
  assert.ok(existsSync(responsivePath), 'responsive.css should exist');
});
