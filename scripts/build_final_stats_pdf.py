"""Builds reports/FINAL_STATS.pdf (all headline results, with the honest verdict). Numbers are copied from reports/*.json at the time of writing; re-check TRACKER.md for updates."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parents[1] / "reports" / "FINAL_STATS.pdf"
ss = getSampleStyleSheet(); H = ParagraphStyle("h", parent=ss["Heading2"], textColor=colors.HexColor("#1f3a5f"), spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=11); C = ParagraphStyle("c", parent=B, fontSize=8, leading=10)


def table(rows, widths):
    data = [[Paragraph(f"<b>{c}</b>" if i == 0 else str(c), C) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fb")]), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


S = [Paragraph("Nifty 50 analyser: final statistics", ss["Title"]),
     Paragraph("Generated 2026-11-26. All figures are measured on held-out or point-in-time data and come from <i>reports/*.json</i>. 329 tests pass; 112 checklist items in TRACKER.md are done.", B),
     Paragraph("Verdict: filing understanding reaches 97&ndash;100% accuracy; tradable stock direction stays at chance (AUC about 0.50&ndash;0.52) at every horizon tested; only the non-tradable opening gap reaches AUC 0.75 (index) / 0.70 (stocks); no statistically established, tradable edge exists.", B)]

S += [Paragraph("1. Understanding filings (high accuracy)", H), table([
    ["Task", "Metric", "Result"],
    ["Event-type classifier (production, test 2025-06 onward)", "Accuracy raw / when it accepts / coverage", "<b>97.3% / 99.45% / 94.3%</b>"],
    ["Same, natural-mix sample", "Accuracy raw / when it accepts", "97.3% / 99.2%"],
    ["14 classes", "Accuracy / macro-F1", "91.9% / 0.929"],
    ["Results and board-outcome merged", "Accuracy / macro-F1", "97.0% / 0.971"],
    ["Earliest conservative version, all rows", "Accuracy / macro-F1", "85.2% / 0.81"],
    ["Date questions, default pipeline, fresh draw", "Accuracy", "<b>100%</b> (120 of 120)"],
    ["Date questions naming a month and year", "Accuracy", "<b>99.1%</b> (107 of 108)"],
    ["Date questions with recency flag, fresh draw", "Accuracy", "99.2% (120)"],
    ["Date questions, earlier draw", "Accuracy", "96.7% (120)"],
    ["Date questions with no recency handling", "Accuracy", "57.5% (80), then 91.2% after the prompt fix"],
    ["Recency boost in retrieval, holdout", "Right filing in top 5 / top 3", "0.90 &rarr; 1.00 / 0.525 &rarr; 0.985"]], [62, 52, 62])]

S += [Paragraph("2. Search and answering (open-ended)", H), table([
    ["Metric", "Result"],
    ["Topical search nDCG@5: old default / routing / reranker / both", "0.665 / 0.693 / 0.756 / <b>0.768</b>"],
    ["Trained cross-encoders vs plain order (nDCG@5)", "MiniLM +0.025, bge-reranker +0.014"],
    ["LLM judges: agreement kappa / my grades vs judges kappa", "0.888 / 0.855"],
    ["/v3/explain answer rate on answerable packs", "46&ndash;50% &rarr; 54% &rarr; <b>76.9%</b> (digest fallback)"],
    ["Same with qwen3:14b, before the digest", "61.5%"],
    ["Unfaithful answers released (judge)", "0"],
    ["Guardrails on planted answers: faithful pass / fabrications blocked", "100% / 95.7%"],
    ["NLI strict mode: fabrications blocked / real faithful answers withheld", "99.9% / 14% (3 of 21)"]], [110, 66])]

S += [Paragraph("3. Predicting stock direction", H), table([
    ["Test", "Result", "Tradable?"],
    ["Tabular models, daily", "AUC about 0.50", "No edge"],
    ["qwen3:14b on 400 announcements", "52.0%, AUC 0.515 (baseline 54.0%)", "No edge"],
    ["FinBERT", "50.3%, AUC 0.508", "No edge"],
    ["Chronos-Bolt, 1 day / 5 days", "50.0% (AUC 0.482) / 52.7% (AUC 0.513)", "No edge"],
    ["Volatility above median, 5 / 20 days", "57.7% (naive 57.1%) / 50.3% (naive 62.5%)", "No"],
    ["Hourly bars, decision 10:15 / 13:15", "51.2% / 51.3%", "No edge"],
    ["US close &rarr; Nifty opening gap: all / &ge;0.5% (54% of days) / &ge;1% (26%) / &ge;1.5% (12%)", "68.4% / 78.1% / 84.5% / 86.1%", "<b>No</b> (gap set at the open)"],
    ["Same &rarr; close-to-close", "58.0% / 64.7% / 68.2% / 72.3%", "No"],
    ["Same &rarr; open-to-close", "46&ndash;48% at every size", "Chance"],
    ["Hourly: from 10:15 / 13:15 (24 instruments)", "51.2% / 51.3% (AUC 0.51)", "No edge"],
    ["<b>Pre-open model</b> (US indices, VIX, dollar, Brent, gold, yields, USD/INR): Nifty opening gap", "AUC <b>0.751</b>, accuracy 68.6% (base 60.7%), top quarter 88.3%", "<b>No</b> (gap set at the open)"],
    ["Same: Nifty close-to-close / open-to-close", "AUC 0.600 / 0.499", "No / chance"],
    ["Same: 59 stocks, gap / close-to-close / open-to-close (v2 features)", "AUC 0.696 / 0.562 / 0.518", "Only gap is predictable"]], [78, 62, 36])]

S += [Paragraph("4. Portfolio and costs", H), table([
    ["Metric", "Result"],
    ["Long/short Sharpe, seed-averaged, 5-day hold: gross / net", "0.45 / &minus;0.69"],
    ["Same, 20-day hold: gross / net", "0.57 / <b>0.14</b> (95% interval &minus;0.76 to +0.89)"],
    ["Deflated Sharpe, 20-day gross, at 10 / 50 / 100 / 500 trials", "0.24 / 0.05 / 0.03 / 0.005"],
    ["Sector-neutral Sharpe, 20-day: gross / net", "&minus;0.09 / &minus;0.58"],
    ["Rank IC", "+0.031 (t 5.6; about 2.5 after overlap correction)"],
    ["Nested hyperparameter tuning gain", "+0.003 &plusmn; 0.007 AUC"],
    ["Half-spread from public prices vs assumed", "27 bps (Corwin&ndash;Schultz), 69 bps (Abdi&ndash;Ranaldo) vs 5 bps"]], [96, 80])]

S += [Paragraph("5. Project status", H), table([
    ["Item", "Status"],
    ["Paper ledgers", "Two arms, hash chains intact, 2 realised days (noise). Next books due about 2026-12-18."],
    ["Still open", "Broker fills; your pass over the label sheet (scripts/label_cli.py); installing the launchd job; calendar time for the ledgers"]], [40, 136]),
      Spacer(1, 4), Paragraph("Why direction stays at chance: daily and hourly prices and public announcements are already reflected in Nifty 50 prices. The only high direction numbers (78&ndash;86%) describe the opening gap, which is set in the pre-open auction after the US move is known and cannot be traded.", B)]

OUT.parent.mkdir(exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm, title="Nifty 50 analyser: final statistics").build(S)
print(OUT)
