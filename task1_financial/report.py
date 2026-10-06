from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISCLAIMER = "Research demonstration only, not investment advice. Data may be delayed or incomplete; LLM reasoning may be incorrect. No trading action is authorized."


def display_number(value, percent=False):
    if value is None:
        return "Unavailable"
    return f"{value:.2%}" if percent else f"{value:,.2f}"


def render_report(result, frame, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = result.summary
    fig, ax = plt.subplots(figsize=(10, 3))
    for column, label in [("Close", "Adjusted close"), ("sma_50", "SMA50"), ("sma_200", "SMA200")]:
        ax.plot(frame.index[-252:], frame[column].iloc[-252:], label=label, linewidth=1.5)
    ax.set_title(f"{summary.ticker}: price and trend")
    ax.set_ylabel(summary.currency or "Price")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140)
    fig.savefig(directory / f"{summary.ticker}_chart.png", dpi=140)
    plt.close(fig)
    top = sorted(result.headlines, key=lambda h: result.sentiments[h.id].confidence
                 if h.id in result.sentiments else -1, reverse=True)[:3]
    snapshot = (f"Latest completed adjusted close: {display_number(summary.latest_close)} {summary.currency or ''}; "
                f"52-week range: {display_number(summary.low_52w)} - {display_number(summary.high_52w)}; "
                f"P/E: {display_number(summary.pe_ratio)}; YTD: {display_number(summary.ytd_return, True)}.")
    technical = f"Momentum: {summary.momentum_signal}. Indicators: " + "; ".join(
        f"{key}={display_number(value)}" for key, value in result.indicators.items())
    recommendation = (f"{result.signal.recommendation}: {result.signal.justification}"
                      if result.signal else "Unavailable: see warnings.")
    score = display_number(result.aggregate_sentiment)
    headlines_text = "\n".join(f"- {h.title} ({h.source})" for h in top)
    markdown = (f"# {summary.ticker} equity research brief\n\nAs of {summary.as_of}\n\n"
                f"## Company snapshot\n{snapshot}\n\n## Technical outlook\n{technical}\n\n"
                f"![Price chart]({summary.ticker}_chart.png)\n\n## News sentiment\n"
                f"Heuristic score: {score}; validated {len(result.sentiments)}/{len(result.headlines)} headlines.\n"
                f"{headlines_text}\n\n## LLM recommendation\n{recommendation}\n\n"
                f"## Limitations\n" + "\n".join(f"- {w}" for w in result.warnings)
                + f"\n\n{DISCLAIMER}\n")
    (directory / f"{summary.ticker}_brief.md").write_text(markdown, encoding="utf-8")
    esc = html.escape
    image = base64.b64encode(buffer.getvalue()).decode()
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(summary.ticker)} research</title>
<style>body{{font:14px/1.45 system-ui;max-width:960px;margin:28px auto;padding:0 24px;color:#17243b}}
h1{{color:#154a6b}}h2{{font-size:17px;margin-bottom:6px}}img{{width:100%}}footer{{border-top:1px solid #ccc;padding-top:12px}}
.meta{{color:#586579}}@media print{{body{{margin:0;font-size:10px}}img{{max-height:230px;object-fit:contain}}}}</style></head>
<body><h1>{esc(summary.ticker)} equity research brief</h1><p class="meta">As of {esc(summary.as_of)}</p>
<h2>Company snapshot</h2><p>{esc(snapshot)}</p><h2>Technical outlook</h2><p>{esc(technical)}</p>
<img alt="Adjusted close, SMA50 and SMA200" src="data:image/png;base64,{image}">
<h2>News sentiment</h2><p>Heuristic score {esc(score)}; {len(result.sentiments)}/{len(result.headlines)} validated.</p>
<ul>{''.join('<li>'+esc(h.title)+' - '+esc(h.source)+'</li>' for h in top)}</ul>
<h2>LLM recommendation</h2><p>{esc(recommendation)}</p>
<h2>Limitations</h2><ul>{''.join('<li>'+esc(w)+'</li>' for w in result.warnings)}</ul>
<footer>{esc(DISCLAIMER)}</footer></body></html>"""
    path = directory / f"{summary.ticker}_brief.html"
    path.write_text(page, encoding="utf-8")
    return path
