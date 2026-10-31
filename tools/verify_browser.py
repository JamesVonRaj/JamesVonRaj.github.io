"""Compare a baseline website with a themed version in desktop/mobile modes.

Serve both directories, then pass their base URLs. Remote artwork is disabled
and Poppins is served from local fonts so unrelated network failures cannot
change the comparison. Requires requirements-theme-dev.txt and Playwright Chromium.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

from site_theme.theme import font_files


async def verify(args):
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = {p.name: p for p in font_files()}
    origins = {urlsplit(url).netloc for url in (args.baseline, args.candidate)}
    report = []
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        for page_name in ('index.html', 'music.html'):
            for width in (1200, 390):
                for accessible in (False, True):
                    captures = []
                    for label, base in (('before', args.baseline), ('after', args.candidate)):
                        context = await browser.new_context(viewport={'width': width, 'height': 900}, reduced_motion='reduce')
                        await context.add_init_script(f"localStorage.setItem('accessible', '{str(accessible).lower()}')")

                        async def route(request):
                            url = request.request.url
                            if 'fonts.googleapis.com' in url:
                                css = '\n'.join(
                                    f"@font-face{{font-family:Poppins;font-weight:{weight};src:url('https://site-theme.local/{name}')}}"
                                    for weight, name in ((400, 'Poppins-Regular.ttf'), (600, 'Poppins-SemiBold.ttf'), (700, 'Poppins-Bold.ttf')))
                                await request.fulfill(body=css, content_type='text/css')
                            elif urlsplit(url).netloc == 'site-theme.local':
                                await request.fulfill(body=fonts[Path(urlsplit(url).path).name].read_bytes(),
                                                      content_type='font/ttf', headers={'Access-Control-Allow-Origin': '*'})
                            elif urlsplit(url).netloc in origins:
                                await request.continue_()
                            else:
                                await request.abort()

                        await context.route('**/*', route)
                        page = await context.new_page()
                        errors = []
                        page.on('pageerror', lambda error: errors.append(str(error)))
                        await page.goto(base.rstrip('/') + '/' + page_name)
                        await page.evaluate('document.fonts.ready')
                        if page_name == 'index.html':
                            await page.locator('.research-card').first.click()
                            await page.evaluate("document.querySelectorAll('video').forEach(v => v.pause())")
                        await page.mouse.move(0, 0)
                        # Clicking a card can scroll it into view. Normalize the
                        # scroll position so fixed controls occupy the same place
                        # in both full-page screenshots.
                        await page.evaluate('window.scrollTo(0, 0)')
                        await page.wait_for_timeout(200)
                        computed = await page.evaluate('''() => [...document.querySelectorAll('body *')]
                            .filter(e => !['SCRIPT','STYLE','SOURCE'].includes(e.tagName)).map(e => {
                            let s = getComputedStyle(e); return {tag:e.tagName, color:s.color, background:s.backgroundColor,
                            border:s.borderColor, shadow:s.boxShadow, width:s.width, height:s.height}; })''')
                        path = args.output / f'{page_name}-{width}-{accessible}-{label}.png'
                        await page.screenshot(path=str(path), full_page=True, animations='disabled')
                        captures.append((computed, np.asarray(Image.open(path)), errors))
                        await context.close()
                    left, right = captures
                    differences = [(i, a, b) for i, (a, b) in enumerate(zip(left[0], right[0])) if a != b]
                    identical = left[1].shape == right[1].shape and np.array_equal(left[1], right[1])
                    report.append({'page': page_name, 'width': width, 'accessible': accessible,
                                   'style_differences': differences, 'pixel_identical': identical,
                                   'errors': left[2] + right[2]})
                    print(page_name, width, accessible, 'pixel identical:', identical, flush=True)
        await browser.close()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if any(not row['pixel_identical'] or row['style_differences'] or row['errors'] for row in report):
        raise SystemExit('Browser comparison found differences; inspect report.json and the screenshots.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', type=Path, default=Path('.theme-build/browser'))
    asyncio.run(verify(parser.parse_args()))
