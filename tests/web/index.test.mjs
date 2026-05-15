// Test: Validate index.html references core frontend assets and contains required elements
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const indexPath = path.join(repoRoot, 'src', 'web', 'index.html');


test('src/web index references the core frontend assets', async () => {
	const html = await readFile(indexPath, 'utf8');

	assert.ok(html.includes('./css/style.css'), 'index.html should reference style.css');
	assert.ok(html.includes('./css/responsive.css'), 'index.html should reference responsive.css');
	assert.ok(html.includes('./js/main.js'), 'index.html should load main.js as a module');
	assert.ok(html.includes('id="predict-form"'), 'index.html should contain the prediction form');
	assert.ok(html.includes('id="result-tab"'), 'index.html should contain the result tab');
	assert.ok(html.includes('id="history-tab"'), 'index.html should contain the history tab');

	assert.ok(existsSync(path.join(repoRoot, 'src', 'web', 'css', 'style.css')), 'style.css should exist');
	assert.ok(existsSync(path.join(repoRoot, 'src', 'web', 'css', 'responsive.css')), 'responsive.css should exist');
	assert.ok(existsSync(path.join(repoRoot, 'src', 'web', 'js', 'main.js')), 'main.js should exist');
});
