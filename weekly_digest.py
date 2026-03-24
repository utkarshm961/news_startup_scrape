"""
weekly_digest.py

Reads Watchlist companies from Notion Sourcing Database,
scrapes last 7 days of news for each, summarizes via Claude API,
and sends a digest email via Gmail.

Usage:
    python weekly_digest.py              # run immediately
    python weekly_digest.py --schedule   # run every Monday at 11am
"""

import os
import sys
import smtplib
import argparse
import html as html_lib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import anthropic
from notion_client import Client



# ── Point to your repo root so src/ imports work ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import Company
from src.news_scraper import scrape_google_news_rss

# ══════════════════════════════════════════════════════════════
# CONFIGURATION — fill these in or set as environment variables
# ══════════════════════════════════════════════════════════════

NOTION_TOKEN       = os.getenv("NOTION_TOKEN", "YOUR_NOTION_INTEGRATION_TOKEN")
NOTION_DATABASE_ID = "5d7fb17768984a1d9117033d567bfe86"  # Sourcing Database

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "YOUR_ANTHROPIC_API_KEY")

# Gmail — use an App Password, not your real Gmail password
# Generate at: myaccount.google.com/apppasswords
GMAIL_SENDER       = "sia.pi.2022@gmail.com"
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "YOUR_GMAIL_APP_PASSWORD")
EMAIL_RECIPIENT    = "sia.pi.2022@gmail.com"   # change to final recipient

NEWS_LOOKBACK_DAYS = 7


# ══════════════════════════════════════════════════════════════
# STEP 1 — Read Watchlist from Notion
# ══════════════════════════════════════════════════════════════

def get_watchlist_companies(notion: Client) -> list:
    """
    Query Sourcing Database filtered to Deal Stage = Watchlist.
    FIX 5: Handles Notion pagination (default cap is 100 results).
    Returns: [{name, dri}]
    """
    companies = []
    start_cursor = None

    while True:
        query_params = {
            "database_id": NOTION_DATABASE_ID,
            "filter": {
                "property": "Deal Stage",
                "multi_select": {"contains": "Watchlist"}
            }
        }
        if start_cursor:
            query_params["start_cursor"] = start_cursor

        response = notion.databases.query(**query_params)

        for page in response.get("results", []):
            props = page.get("properties", {})

            # FIX 1: Find title property dynamically — don't guess the column name
            name = ""
            for prop_name, prop_value in props.items():
                if prop_value.get("type") == "title":
                    title_list = prop_value.get("title", [])
                    if title_list:
                        name = title_list[0].get("plain_text", "").strip()
                    break

            if not name:
                print(f"  [WARN] Could not extract name from page {page.get('id', '?')}. "
                      f"Properties: {list(props.keys())}")
                continue

            # DRI — people property
            dri = "Unassigned"
            people = props.get("DRI", {}).get("people", [])
            if people:
                dri = people[0].get("name", "Unassigned")

            companies.append({"name": name, "dri": dri})

        # Stop if no more pages
        if not response.get("has_more"):
            break
        start_cursor = response.get("next_cursor")

    print(f"[Notion] Found {len(companies)} Watchlist companies")
    return companies


# ══════════════════════════════════════════════════════════════
# STEP 2 — Scrape + Summarize news for each company
# ══════════════════════════════════════════════════════════════

def get_news_summary(company_name: str, claude: anthropic.Anthropic) -> dict:
    """
    Scrape last 7 days of news and summarize via Claude.
    Returns: {has_news, summary, urls, article_count}
    """
    company = Company(
        name=company_name,
        founding_year="N/A",
        sector="VC Watchlist",
        founders="N/A",
        description=f"VC watchlist company: {company_name}",
        company_type="watchlist",
        search_terms=[company_name],
    )

    raw_articles = scrape_google_news_rss(company, last_n_days=NEWS_LOOKBACK_DAYS)
    articles = [a.to_dict() for a in raw_articles]

    if not articles:
        return {
            "has_news": False,
            "summary": "No news this week.",
            "urls": [],
            "article_count": 0
        }

    # Build article text for Claude (cap at 10 articles)
    articles_text = ""
    urls = []
    for i, a in enumerate(articles[:10], 1):
        articles_text += (
            f"{i}. {a['title']}\n"
            f"   Source: {a.get('source', 'Unknown')} | "
            f"Date: {a.get('published_date', 'Unknown')}\n"
            f"   {a.get('snippet', '')}\n\n"
        )
        if a.get("url"):
            urls.append({"title": a["title"], "url": a["url"]})

    prompt = f"""You are a VC analyst assistant at an early-stage Indian venture capital firm.

Summarize the following news articles about {company_name} in 3-4 concise bullet points.
Focus only on what is investment-relevant: funding rounds, product launches, partnerships,
revenue/growth signals, regulatory issues, leadership changes, or competitive moves.
Be factual, specific, and brief. Skip generic filler.

Articles:
{articles_text}

Respond with bullet points only. No intro, no outro."""

    message = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    # FIX 7: Guard against empty Claude response
    if not message.content or not hasattr(message.content[0], "text"):
        summary = "Summary unavailable — Claude returned an empty response."
    else:
        summary = message.content[0].text.strip() or "Summary unavailable."
    return {
        "has_news": True,
        "summary": summary,
        "urls": urls[:5],
        "article_count": len(articles)
    }


# ══════════════════════════════════════════════════════════════
# STEP 3 — Send Gmail digest
# ══════════════════════════════════════════════════════════════

def send_gmail_digest(results: list):
    """Sends HTML digest email via Gmail SMTP."""
    today = datetime.now().strftime("%d %B %Y")

    with_news    = [r for r in results if r["data"]["has_news"]]
    without_news = [r for r in results if not r["data"]["has_news"]]

    html = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;
                 max-width: 680px; margin: auto; padding: 20px;">

      <div style="background: linear-gradient(135deg, #1a1a2e, #0f3460);
                  border-radius: 12px; padding: 28px; margin-bottom: 24px;">
        <h1 style="color: #4ECDC4; margin: 0; font-size: 24px;">
          📰 Newssyy — Weekly Watchlist Digest
        </h1>
        <p style="color: #aaa; margin: 8px 0 0;">
          {today} &nbsp;|&nbsp;
          {len(with_news)} companies with news &nbsp;|&nbsp;
          {len(without_news)} quiet this week
        </p>
      </div>
    """

    if with_news:
        html += "<h2 style='color:#1a1a2e;'>📈 Companies In The News</h2>"
        for item in with_news:
            # FIX 6: Escape special characters to prevent broken HTML
            safe_company = html_lib.escape(item['company'])
            safe_summary = html_lib.escape(item['data']['summary'])
            safe_dri     = html_lib.escape(item['dri'])
            html += f"""
            <div style="border:1px solid #e0e0e0; border-radius:8px;
                        padding:16px; margin-bottom:16px;">
              <h3 style="margin:0 0 4px; color:#0f3460;">
                {safe_company}
                <span style="font-size:12px; background:#4ECDC4; color:white;
                             padding:2px 8px; border-radius:10px; margin-left:8px;">
                  {item['data']['article_count']} articles
                </span>
              </h3>
              <p style="color:#888; font-size:13px; margin:0 0 10px;">
                DRI: {safe_dri}
              </p>
              <div style="white-space:pre-line; line-height:1.7;">
                {safe_summary}
              </div>
            """
            if item["data"]["urls"]:
                html += "<p style='margin:12px 0 4px;'><strong>Sources:</strong></p><ul>"
                for link in item["data"]["urls"]:
                    safe_title = html_lib.escape(link["title"])
                    html += (
                        f'<li><a href="{link["url"]}" style="color:#0f3460;">'
                        f'{safe_title}</a></li>'
                    )
                html += "</ul>"
            html += "</div>"

    if without_news:
        names = html_lib.escape(", ".join(r["company"] for r in without_news))
        html += f"""
        <div style="background:#f9f9f9; border-radius:8px; padding:14px; margin-top:8px;">
          <strong>🔕 No news this week:</strong>
          <span style="color:#888;"> {names}</span>
        </div>
        """

    html += """
      <hr style="margin-top:32px;">
      <p style="color:#bbb; font-size:12px;">
        Auto-generated by Newssyy every Monday at 11am. Coverage: last 7 days.
      </p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Newssyy] Weekly Watchlist Digest — {today}"
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = EMAIL_RECIPIENT
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())

    print(f"[Email] Digest sent to {EMAIL_RECIPIENT}")


# ══════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════════

def validate_config():
    """
    FIX 2: Validate all credentials before doing any work.
    Crashes immediately with a clear message instead of failing mid-run.
    """
    errors = []
    if NOTION_TOKEN == "YOUR_NOTION_INTEGRATION_TOKEN":
        errors.append("NOTION_TOKEN not set — run: export NOTION_TOKEN=your_token")
    if ANTHROPIC_API_KEY == "YOUR_ANTHROPIC_API_KEY":
        errors.append("ANTHROPIC_API_KEY not set — run: export ANTHROPIC_API_KEY=your_key")
    if GMAIL_APP_PASSWORD == "YOUR_GMAIL_APP_PASSWORD":
        errors.append(
            "GMAIL_APP_PASSWORD not set.\n"
            "    1. Enable 2FA on your Google account\n"
            "    2. Go to myaccount.google.com/apppasswords\n"
            "    3. Generate App Password for 'Mail'\n"
            "    4. Run: export GMAIL_APP_PASSWORD=the_16_char_code"
        )
    if errors:
        print("\n[CONFIG ERROR] Fix these before running:\n")
        for e in errors:
            print(f"  • {e}\n")
        sys.exit(1)


def run_weekly_digest():
    print(f"\n{'='*52}")
    print(f"  Newssyy Weekly Digest — {datetime.now().strftime('%d %b %Y %H:%M')}")
    print(f"{'='*52}\n")

    # FIX 2: Validate credentials before doing anything
    validate_config()

    notion = Client(auth=NOTION_TOKEN)
    claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # 1. Pull Watchlist from Notion
    companies = get_watchlist_companies(notion)
    if not companies:
        print("[WARN] No Watchlist companies found. Exiting.")
        return

    # 2. Scrape + summarize each company
    # FIX 4: Wrap each company in try/except so one failure doesn't kill the whole run
    results = []
    failed = []
    for c in companies:
        print(f"[Processing] {c['name']}...")
        try:
            data = get_news_summary(c["name"], claude)
            status = f"{data['article_count']} articles found" if data["has_news"] else "No news"
            print(f"  -> {status}")
            results.append({"company": c["name"], "dri": c["dri"], "data": data})
        except Exception as e:
            print(f"  [ERROR] Failed to process {c['name']}: {e}")
            failed.append(c["name"])
            # Add a placeholder so it still appears in the email
            results.append({
                "company": c["name"],
                "dri": c["dri"],
                "data": {
                    "has_news": False,
                    "summary": f"⚠️ Processing error this week: {str(e)[:100]}",
                    "urls": [],
                    "article_count": 0
                }
            })

    if failed:
        print(f"\n[WARN] {len(failed)} companies failed: {', '.join(failed)}")

    # 3. Send email
    send_gmail_digest(results)
    print(f"\n[Done] Weekly digest complete.\n")


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on schedule every Monday at 11:00 AM"
    )
    args = parser.parse_args()

    if args.schedule:
        import schedule
        import time
        schedule.every().monday.at("11:00").do(run_weekly_digest)
        print("Scheduler active — fires every Monday at 11:00 AM")
        print("Press Ctrl+C to stop\n")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_weekly_digest()
