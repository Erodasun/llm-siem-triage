# 46_dashboard_screenshots.py
# Снимки дашборда для отчёта (reports/figures/dash_*.png). Запускать после 17_stage6_flask_app.py --static-only.
# Нужен playwright с Chromium: pip install playwright && playwright install chromium
#   dash_1_overview.png - общий вид страницы: список конфигураций слева, результат справа
#   dash_1c_choose.png  - блок «Какой тандем выбрать»
#   dash_1b_card.png    - результат Тандема 2 в режиме сравнения с LightGBM
#   dash_2_charts.png   - сравнение со всеми конфигурациями: F1 с доверительными интервалами
#   dash_2b_load.png    - сравнение со всеми конфигурациями: доля отмеченных как атака и приоритеты
#   dash_3_queue.png    - разбор алертов: список и объяснение выбранного алерта
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "results" / "stage_6" / "dashboard.html").as_uri()
FIG = ROOT / "reports" / "figures"
W = 1280


def clip(pg, sel, path, pad=6, h=None):
    box = pg.evaluate(f"(()=>{{var r=document.querySelector('{sel}').getBoundingClientRect();"
                      f"return [r.left+scrollX,r.top+scrollY,r.width,r.height]}})()")
    pg.screenshot(path=str(path), full_page=True,
                  clip={"x": max(0, box[0] - pad), "y": max(0, box[1] - pad), "width": box[2] + 2 * pad,
                        "height": (h or box[3]) + 2 * pad})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": 900}, device_scale_factor=2)
        pg.goto(HTML + "#id=tm_gpt&cmp=&tab=p1")
        pg.wait_for_timeout(800)
        pg.add_style_tag(content="aside,.detail{position:static!important;max-height:none!important}")
        # 1: общий вид (шапка, список конфигураций, результат)
        bottom = pg.evaluate("Math.max(document.querySelector('aside').getBoundingClientRect().bottom,"
                             "document.getElementById('sel').getBoundingClientRect().bottom)+scrollY+12")
        pg.screenshot(path=str(FIG / "dash_1_overview.png"), full_page=True, clip={"x": 0, "y": 0, "width": W, "height": bottom})
        # 1c: выбор лучшего тандема
        clip(pg, ".tcmp", FIG / "dash_1c_choose.png")
        # 2: результат в режиме сравнения
        pg.select_option("#cmp", "lgb")
        pg.wait_for_timeout(200)
        clip(pg, "#sel", FIG / "dash_1b_card.png")
        # 3: графики сравнения со всеми
        clip(pg, "#s-charts", FIG / "dash_2_charts.png")
        pg.click("#ctabs button[data-p=p4]")
        pg.wait_for_timeout(200)
        clip(pg, "#s-charts", FIG / "dash_2b_load.png")
        # 4: разбор алертов (видимая часть списка и объяснение)
        det_h = pg.evaluate("document.getElementById('detail').getBoundingClientRect().bottom-document.getElementById('qsec').getBoundingClientRect().top+18")
        clip(pg, "#qsec", FIG / "dash_3_queue.png", h=max(det_h, 520))
        b.close()
    print("ok:", FIG)


if __name__ == "__main__":
    sys.exit(main())
