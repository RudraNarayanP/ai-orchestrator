"""ONE-OFF (user-authorised 2026-10-05): dismiss pi.ai's info dialog 'Memory just got better ... Continue to Pi'.
Not part of the adapter; the adapter itself never clicks dialogs. Clicks only a button whose text is exactly
'Continue to Pi' inside a dialog that contains 'Memory just got better'."""
import asyncio, sys, json
sys.path.insert(0, ".")
from backend.browser.factory import create_engine
from backend.settings import load_settings

JS = """() => {
  const dlgs = [...document.querySelectorAll('[role=dialog],[aria-modal=true]')].filter(d => /Memory just got better/i.test(d.innerText || ''));
  if (dlgs.length !== 1) return JSON.stringify({clicked: false, why: 'dialogs found: ' + dlgs.length});
  const btns = [...dlgs[0].querySelectorAll('button')].filter(b => (b.innerText || '').trim() === 'Continue to Pi');
  if (btns.length !== 1) return JSON.stringify({clicked: false, why: 'buttons found: ' + btns.length});
  btns[0].click();
  return JSON.stringify({clicked: true});
}"""

async def main():
    s = load_settings(); s.browser.driver = "chrome_use"
    eng = create_engine(s); await eng.start()
    try:
        page = await eng.open_research_page("pi", s.providers["pi"].url, key="pi_oneoff")
        await page.wait_for_timeout(4000)
        print("url", page.url)
        print(await page.evaluate(JS))
        await page.wait_for_timeout(2000)
        print("dialog still there:", await page.evaluate("() => !![...document.querySelectorAll('[role=dialog]')].find(d => /Memory just got better/i.test(d.innerText||''))"))
    finally:
        await eng.stop(keep_windows=False)
asyncio.run(main())
