import { test, expect } from '@playwright/test';

const BASE = process.env.BASE_URL || 'http://localhost:8675';
const DATASET = 'test_smoke_eval';

import { removeNextDevOverlay } from './test_helpers';

test.describe('Dataset evaluation UI', () => {
  test('opens dataset page, opens eval modal, enqueues job and lists it', async ({ page }) => {
    const url = `${BASE}/datasets/${DATASET}`;
    const info = test.info();

    await page.goto(url, { waitUntil: 'networkidle' });

    // Save HTML snapshot after navigation for debugging
    const beforeHtml = await page.content();
    const fs = require('fs');
    await fs.promises.writeFile(info.outputPath('dataset_before.html'), beforeHtml);

    // Basic checks
    await expect(page.locator(`text=Dataset: ${DATASET}`)).toBeVisible({ timeout: 10000 });

    // Remove dev overlay if present so it doesn't intercept pointer events
    await removeNextDevOverlay(page);

    // Save HTML snapshot after overlay removal
    const afterOverlayHtml = await page.content();
    await fs.promises.writeFile(info.outputPath('dataset_after_overlay.html'), afterOverlayHtml);

    await expect(page.locator('text=Run Evaluation')).toBeVisible();

    // Open modal (force click to bypass dev overlay if present)
    await page.locator('text=Run Evaluation').waitFor({ state: 'visible', timeout: 10000 });
    // Use DOM click in case synthetic click is blocked by overlay/portal
    await page.locator('text=Run Evaluation').evaluate((el: any) => el && (el as HTMLElement).click());
    await expect(page.locator('text=Run Dataset Evaluation')).toBeVisible({ timeout: 10000 });

    // Click Run with defaults (use role to avoid ambiguous matches)
    const runButton = page.getByRole('button', { name: 'Run', exact: true });
    await runButton.waitFor({ state: 'visible', timeout: 10000 });
    await runButton.click({ force: true });

    // Wait for modal to close OR show an error message
    const errLocator = page.locator('.text-sm.text-red-600');
    const closed = await Promise.race([
      page.waitForSelector('text=Run Dataset Evaluation', { state: 'hidden', timeout: 10000 }).then(() => ({ closed: true })),
      errLocator.waitFor({ state: 'visible', timeout: 10000 }).then(() => ({ closed: false })),
    ]);

    if (!closed || !closed.closed) {
      const msg = (await errLocator.textContent()) || 'Unknown error';
      throw new Error('Eval enqueue failed: ' + msg.trim());
    }

    // Wait for jobs list to refresh
    await page.waitForSelector('text=Recent Evaluations', { timeout: 10000 });

    // Poll for the queued job to appear
    const jobRow = page.locator('table >> text=queued');
    await expect(jobRow.first()).toBeVisible({ timeout: 20000 });

    // Ensure there's a View JSON link (API-backed)
    await expect(page.locator('text=View JSON').first()).toBeVisible();
  });
});