import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";

const baseURL = process.env.RSTOKEN_DEMO_URL || "http://127.0.0.1:7860";
const executablePath = process.env.PLAYWRIGHT_CHROME || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const output = path.resolve("..", ".runtime", "screenshots");
await mkdir(output, { recursive: true });

const browser = await chromium.launch({ executablePath, headless: true });
const errors = [];

async function audit(name, viewport) {
  const page = await browser.newPage({ viewport, deviceScaleFactor: 1 });
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(`${name}: ${message.text()}`);
  });
  page.on("pageerror", (error) => errors.push(`${name}: ${error.message}`));
  await page.goto(baseURL, { waitUntil: "networkidle", timeout: 180_000 });
  await page.locator(".health.ready").waitFor({ timeout: 180_000 });
  await page.getByRole("button", { name: "执行传输" }).click();
  await page.locator(".result-pane img").waitFor({ timeout: 180_000 });
  await page.locator(".primary-action").filter({ hasText: "执行传输" }).waitFor({ timeout: 180_000 });

  const overflow = await page.evaluate(() => ({
    documentWidth: document.documentElement.scrollWidth,
    viewportWidth: document.documentElement.clientWidth,
    bodyWidth: document.body.scrollWidth,
  }));
  if (overflow.documentWidth > overflow.viewportWidth + 2) {
    errors.push(`${name}: horizontal overflow ${JSON.stringify(overflow)}`);
  }

  await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true });

  if (name.startsWith("desktop")) {
    await page.route("**/api/infer", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 700));
      await route.continue();
    });
    await page.getByRole("button", { name: "启动动态链路" }).click();
    await page.locator(".link-update").waitFor({ timeout: 30_000 });
    if (!(await page.locator(".result-pane img").isVisible())) {
      errors.push(`${name}: previous reconstruction disappeared during background link update`);
    }
    if ((await page.locator(".image-loading").count()) > 0) {
      errors.push(`${name}: blocking image overlay appeared during dynamic link update`);
    }
    await page.locator(".quality-orbit.k4").waitFor({ timeout: 30_000 });
    await page.screenshot({ path: path.join(output, `${name}-live.png`), fullPage: true });
    await page.getByRole("button", { name: "暂停动态链路" }).click();
  }
  await page.close();
}

await audit("desktop-1440", { width: 1440, height: 1000 });
await audit("mobile-390", { width: 390, height: 844 });
await browser.close();

if (errors.length) {
  throw new Error(`Visual audit failed:\n${errors.join("\n")}`);
}
console.log(`Visual audit passed. Screenshots: ${output}`);
