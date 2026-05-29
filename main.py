from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"])

WB_BASE = "https://api.worldbank.org/v2"

CATALOG = {
    "Indicatori Macroeconomici": [
        {"id": "FR.INR.RINR",       "name": "Dobânda de referință — Rata reală a dobânzii (%)"},
        {"id": "NY.GDP.MKTP.KD.ZG", "name": "GDP Growth Rate — Rata de creștere PIB (%)"},
        {"id": "NY.GDP.MKTP.KD.ZG", "name": "GDP Annual Growth Rate — Creștere PIB anuală (%)"},
        {"id": "FP.CPI.TOTL.ZG",    "name": "Rata inflației — CPI (%)"},
        {"id": "SL.UEM.TOTL.ZS",    "name": "Rata șomajului — Total (%)"},
    ]
}

COUNTRY_CODES = {
    "Romania":"RO","Germany":"DE","France":"FR","Italy":"IT","Spain":"ES",
    "Netherlands":"NL","Poland":"PL","Sweden":"SE","Austria":"AT","Belgium":"BE",
    "Portugal":"PT","Greece":"GR","Hungary":"HU","Czech Republic":"CZ",
    "Bulgaria":"BG","Croatia":"HR","Denmark":"DK","Finland":"FI","Norway":"NO",
    "Switzerland":"CH","United States":"US","United Kingdom":"GB","China":"CN",
    "Japan":"JP","India":"IN","Brazil":"BR","Turkey":"TR","South Korea":"KR",
    "Euro Area":"XC","European Union":"EU","World":"WLD",
}

COUNTRIES_BY_REGION = {
    "Europa": ["Romania","Germany","France","Italy","Spain","Netherlands","Poland",
               "Sweden","Austria","Belgium","Portugal","Greece","Hungary",
               "Czech Republic","Bulgaria","Croatia","Denmark","Finland","Norway","Switzerland"],
    "America & Asia": ["United States","United Kingdom","China","Japan",
                       "India","Brazil","Turkey","South Korea"],
    "Global": ["Euro Area","European Union","World"],
}

def fetch_wb(indicator, country_code):
    url = f"{WB_BASE}/country/{country_code}/indicator/{indicator}?format=json&per_page=100&date=1990:2024"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        payload = r.json()
        if len(payload) < 2 or not payload[1]:
            return []
        records = []
        for item in payload[1]:
            if item.get("value") is not None:
                records.append({"date": str(item["date"]),
                                "value": round(float(item["value"]), 4)})
        return sorted(records, key=lambda x: x["date"])
    except Exception as e:
        print(f"WB error: {e}")
        return []

@app.get("/catalog")
def get_catalog():
    return {"indicators": CATALOG, "countries": COUNTRIES_BY_REGION,
            "country_codes": COUNTRY_CODES}

@app.get("/indicator")
def get_indicator(indicator: str, countries: str = "Romania"):
    country_list = [c.strip() for c in countries.split(",") if c.strip()]
    result = {}
    for country in country_list[:8]:
        code = COUNTRY_CODES.get(country)
        if not code:
            continue
        records = fetch_wb(indicator, code)
        if records:
            result[country] = records
    if not result:
        return {"data": {}, "stats": {}, "latest": {}}
    stats = {}
    latest = {}
    for country, records in result.items():
        vals = [r["value"] for r in records]
        if vals:
            stats[country] = {"mean": round(sum(vals)/len(vals), 4),
                              "max": round(max(vals), 4),
                              "min": round(min(vals), 4)}
            last = sorted(records, key=lambda x: x["date"], reverse=True)[0]
            latest[country] = {"value": last["value"], "date": last["date"]}
    return {"data": result, "stats": stats, "latest": latest}

@app.post("/ai-chat")
async def ai_chat(payload: dict):
    messages       = payload.get("messages", [])
    countries      = payload.get("countries", [])
    indicator_name = payload.get("indicator_name", "")
    stats          = payload.get("stats", {})
    latest         = payload.get("latest", {})

    if not messages:
        return {"reply": "Nu am primit niciun mesaj."}

    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return {"reply": "Cheia OpenAI nu este configurata."}

    countries_str = ", ".join(countries) if countries else "tari diverse"
    latest_text = "\n".join(
        f"  {c}: {i.get('value','-')}% (an {i.get('date','-')})"
        for c, i in latest.items()) or "Nu sunt date."
    stats_text = "\n".join(
        f"  {c}: Medie={s.get('mean')}, Max={s.get('max')}, Min={s.get('min')}"
        for c, s in stats.items())

    system_prompt = (
        f"Esti un expert in economie macroeconomica. "
        f"Analizeaza '{indicator_name}' pentru: {countries_str}.\n"
        f"Valori recente:\n{latest_text}\n"
        f"Statistici:\n{stats_text}\n"
        f"Raspunde in romana, concis."
    )

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": "gpt-4o-mini", "max_tokens": 1000,
                  "messages": [{"role": "system", "content": system_prompt}] + messages},
            timeout=30
        )
        data = response.json()
        return {"reply": data["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"reply": f"Eroare: {str(e)}"}
