import { Page } from '@playwright/test';

export async function removeNextDevOverlay(page: Page) {
  await page.evaluate(() => {
    try {
      // Remove Next dev overlay script-based portal
      const overlay = document.querySelector('[data-nextjs-dev-overlay]');
      if (overlay && overlay.parentElement) {
        overlay.parentElement.removeChild(overlay);
      }
      // Remove any nextjs-portal nodes which may intercept pointer events
      const portals = Array.from(document.querySelectorAll('nextjs-portal'));
      portals.forEach((p) => p.remove());
      // Also clear pointer-events on body as a fallback
      (document.body as HTMLElement).style.pointerEvents = 'auto';
    } catch (e) {
      // ignore
    }
  });
}
