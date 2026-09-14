"""Monthly Meta report preparation. Network access is always read-only.

The live field/action mapping must be reconciled with Ads Manager before use.
No token, API response, or customer data is written to application logs.
"""
import argparse
import calendar
import json
import os
import re
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from zoneinfo import ZoneInfo


ACCOUNT_ID = "158472333524796"
SPREADSHEET_ID = "1dw5ssrZu9UfzymB7GLs0rqZf0LggvKAnC5Tek3go1cM"
HEADERS = ["mês_ano", "plataforma", "destino", "objetivo", "criativo_anúncio",
           "valor_usado", "alcance", "impressões", "resultados", "custo por resultados",
           "cliques", "ctr", "cpm", "visitas ao perfil do instagram", "oportunidades", "negócios"]
PLATFORMS = {"facebook": "Facebook", "instagram": "Instagram"}
DESTINATIONS = {"Conversas": "Whatsapp", "Visitas ao Perfil do Instagram": "Instagram",
                "Lead Site": "Site", "Lead Formulário": "Formulário Meta"}
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


class ReportError(Exception):
    """Safe, non-secret error suitable for CI logs."""


def month_bounds(month=None, today=None):
    today = today or datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    if month is None:
        first = today.replace(day=1)
        previous = date.fromordinal(first.toordinal() - 1)
        month = previous.strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ReportError("O mês deve usar YYYY-MM.")
    try:
        year, number = map(int, month.split("-"))
        start = date(year, number, 1)
        end = date(year, number, calendar.monthrange(year, number)[1])
    except ValueError:
        raise ReportError("Mês inválido.") from None
    if end >= today:
        raise ReportError("Use somente um mês completo já encerrado.")
    return start.isoformat(), end.isoformat()


def numeric(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ReportError("Métrica inválida na resposta da Meta.") from None
    if not result.is_finite() or result < 0:
        raise ReportError("Métrica negativa ou não finita na resposta da Meta.")
    return result


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class MetaClient:
    def __init__(self, token, version):
        if not token or not re.fullmatch(r"v\d+\.\d+", version):
            raise ReportError("Configure META_ACCESS_TOKEN e META_API_VERSION.")
        self.token = token
        self.base = "https://graph.facebook.com/" + version
        self.opener = build_opener(NoRedirect())

    def get(self, path, params):
        if not re.fullmatch(r"(?:act_)?\d+(?:/insights)?", path):
            raise ReportError("Endpoint Meta não permitido.")
        params = {k: json.dumps(v) if isinstance(v, (list, dict, bool)) else v
                  for k, v in params.items()}
        url = self.base + "/" + path + "?" + urlencode(params)
        for attempt in range(4):
            try:
                request = Request(url, headers={"Authorization": "Bearer " + self.token})
                with self.opener.open(request, timeout=60) as response:
                    result = json.load(response)
                if not isinstance(result, dict) or "error" in result:
                    raise ReportError("Resposta inesperada da Meta.")
                return result
            except HTTPError as error:
                # Do not print request URLs, response bodies, or token-containing exceptions.
                status = error.code
                try:
                    detail = json.load(error).get("error", {})
                except (ValueError, AttributeError):
                    detail = {}
                retryable = status == 429 or status >= 500 or detail.get("is_transient") is True
                retryable = retryable or detail.get("code") in (4, 17, 32, 613, 80000)
                if not retryable or attempt == 3:
                    raise ReportError(f"Consulta Meta falhou (HTTP {status}); verifique acesso e campos.") from None
            except (URLError, TimeoutError):
                if attempt == 3:
                    raise ReportError("Falha de conexão com a Meta; nenhuma escrita realizada.") from None
            except ValueError:
                raise ReportError("Resposta Meta não é JSON válido.") from None
            time.sleep(2 ** (attempt + 1))

    def pages(self, path, params):
        params = dict(params)
        seen = set()
        result = []
        for _ in range(1000):
            page = self.get(path, params)
            data = page.get("data")
            if not isinstance(data, list):
                raise ReportError("Página de relatório inválida.")
            result.extend(data)
            paging = page.get("paging", {})
            if not paging.get("next"):
                return result
            cursor = paging.get("cursors", {}).get("after")
            if not cursor or cursor in seen:
                raise ReportError("Paginação incompleta ou repetida; relatório descartado.")
            seen.add(cursor)
            params["after"] = cursor  # Never follow arbitrary next URLs with the token.
        raise ReportError("Limite de paginação atingido; relatório incompleto.")


def collect(client, month, today=None):
    start, end = month_bounds(month, today)
    account = client.get("act_" + ACCOUNT_ID, {"fields": "id,account_id,name,currency,timezone_name"})
    if str(account.get("account_id")) != ACCOUNT_ID or account.get("currency") != "BRL":
        raise ReportError("Conta ou moeda divergente da NEXTQS Brasil.")
    common = {"time_range": {"since": start, "until": end}, "time_increment": "all_days",
              "use_unified_attribution_setting": True, "action_report_time": "impression", "limit": 100}
    fields = "account_id,ad_id,ad_name,adset_id,adset_name,campaign_id,campaign_name,date_start,date_stop,spend,reach,impressions,inline_link_clicks,instagram_profile_visits,actions"
    rows = client.pages("act_" + ACCOUNT_ID + "/insights", dict(common, level="ad", breakdowns=["publisher_platform"], fields=fields))
    totals = client.pages("act_" + ACCOUNT_ID + "/insights", dict(common, level="account", fields="spend,reach,impressions,date_start,date_stop"))
    ads = {}
    adsets = {}
    for row in rows:
        ad_id = str(row["ad_id"])
        if ad_id not in ads:
            ads[ad_id] = client.get(ad_id, {"fields": "id,name,creative{id,name,instagram_permalink_url,effective_object_story_id}"})
        adset_id = str(row["adset_id"])
        if adset_id not in adsets:
            adsets[adset_id] = client.get(adset_id, {"fields": "id,name,destination_type,optimization_goal,attribution_spec,promoted_object"})
    return {"month": start[:7], "account": account, "rows": rows, "totals": totals,
            "ads": ads, "adsets": adsets, "action_report_time": "impression",
            "collected_at": datetime.now(ZoneInfo("UTC")).isoformat()}


def action_value(row, action_type):
    entries = [a for a in row.get("actions", []) if a.get("action_type") == action_type]
    if len(entries) > 1:
        raise ReportError("Evento duplicado na resposta; não somar eventos sobrepostos.")
    return numeric(entries[0]["value"]) if entries else Decimal(0)


def metric(row, spec):
    if not isinstance(spec, dict) or set(spec) not in ({"field"}, {"action_type"}):
        raise ReportError("Mapeamento de métrica não validado.")
    if "action_type" in spec:
        return action_value(row, spec["action_type"])
    if spec["field"] not in row:
        raise ReportError("Campo de métrica ausente; não substituir por zero.")
    return numeric(row[spec["field"]])


def report_rows(raw, mapping):
    """Require a verified mapping per ad set/ad. Never guess what Results means."""
    if raw.get("account", {}).get("account_id") != ACCOUNT_ID:
        raise ReportError("Relatório de outra conta.")
    rows = raw["rows"]
    if not rows:
        raise ReportError("Relatório vazio; preservar planilha existente.")
    expected = sum((numeric(r["spend"]) for r in raw["totals"]), Decimal(0))
    actual = sum((numeric(r["spend"]) for r in rows), Decimal(0))
    if abs(expected - actual) > Decimal("0.02"):
        raise ReportError("Investimento por anúncio não confere com o total da conta.")
    output = []
    keys = set()
    for row in rows:
        if row.get("date_start", "")[:7] != raw["month"] or row.get("date_stop", "")[:7] != raw["month"]:
            raise ReportError("Resposta contém período inesperado.")
        rule = mapping.get("ads", {}).get(row["ad_id"]) or mapping.get("adsets", {}).get(row["adset_id"])
        if not rule or not rule.get("validated"):
            raise ReportError("Há anúncio/conjunto sem classificação validada; escrita bloqueada.")
        objective = rule["objetivo"]
        if objective not in DESTINATIONS:
            raise ReportError("Objetivo sem correspondência no dashboard.")
        platform = PLATFORMS.get(row.get("publisher_platform"))
        if not platform:
            raise ReportError("Plataforma nova exige revisão; não descartar investimento.")
        creative = raw["ads"][row["ad_id"]].get("creative", {})
        url = rule.get("criativo_anuncio") or creative.get("instagram_permalink_url")
        parsed = urlparse(url or "")
        if parsed.scheme != "https" or parsed.hostname not in ("instagram.com", "www.instagram.com", "facebook.com", "www.facebook.com"):
            raise ReportError("Link do criativo ausente ou inválido; revisão necessária.")
        result = metric(row, rule["result_metric"])
        visits = metric(row, rule["profile_visits_metric"])
        spend, impressions = numeric(row["spend"]), numeric(row["impressions"])
        clicks, reach = numeric(row.get("inline_link_clicks", 0)), numeric(row["reach"])
        values = [raw["month"], platform, DESTINATIONS[objective], objective, url,
                  float(spend), float(reach), float(impressions), float(result),
                  float(spend / result) if result else 0, float(clicks),
                  float(clicks / impressions) if impressions else 0,
                  float(spend * 1000 / impressions) if impressions else 0, float(visits)]
        key = tuple(values[:5])
        if key in keys:
            raise ReportError("Vários anúncios compartilham a mesma linha; validar alcance antes de consolidar.")
        keys.add(key)
        output.append(values)
    return sorted(output, key=lambda r: tuple(r[:5]))


def plan_updates(existing, proposed, month):
    """Only touch A:N. Columns O:P and any extra columns stay in their original rows."""
    if not existing or existing[0][:16] != HEADERS:
        raise ReportError("Cabeçalhos da planilha mudaram; nenhuma escrita permitida.")
    lookup = {}
    for index, row in enumerate(existing[1:], 2):
        if row and row[0] == month:
            key = tuple((row + [""] * 5)[:5])
            if key in lookup:
                raise ReportError("Linhas duplicadas no mês existente; revisão necessária.")
            lookup[key] = index
    incoming = {tuple(row[:5]) for row in proposed}
    if len(incoming) != len(proposed) or not proposed or any(len(r) != 14 or r[0] != month for r in proposed):
        raise ReportError("Proposta vazia, duplicada ou de outro mês.")
    if set(lookup) - incoming:
        raise ReportError("Há linhas históricas sem correspondência; não remover automaticamente.")
    last = max((i for i, row in enumerate(existing, 1) if any(str(v).strip() for v in row)), default=1)
    changes = []
    for row in proposed:
        index = lookup.get(tuple(row[:5]))
        if index is None:
            last += 1
            index = last
        changes.append({"range": f"A{index}:N{index}", "values": [row]})
    return changes


def load_sheet_rows(credentials_json):
    """Read unformatted values from the existing worksheet without changing it."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        info = json.loads(credentials_json or "")
        credentials = Credentials.from_service_account_info(info, scopes=GOOGLE_SCOPES)
        worksheet = gspread.authorize(credentials).open_by_key(SPREADSHEET_ID).worksheet("meta_campanhas")
        return worksheet.get("A1:P", value_render_option="UNFORMATTED_VALUE")
    except (ValueError, TypeError, KeyError):
        raise ReportError("Credencial Google inválida; nenhuma escrita realizada.") from None
    except Exception:
        # Provider exceptions may contain request details. Keep CI logs generic.
        raise ReportError("Não foi possível ler meta_campanhas com a conta de serviço.") from None


def sheet_number(value):
    if isinstance(value, bool) or value is None:
        raise ReportError("Valor numérico inválido na planilha.")
    if isinstance(value, (int, float, Decimal)):
        return numeric(value)
    text = str(value).strip().replace("\u00a0", "").replace("R$", "").replace("%", "")
    if not text:
        return Decimal(0)
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    return numeric(text)


def sheet_month(value):
    """Normalize either a displayed YYYY-MM value or a Google Sheets date serial."""
    if isinstance(value, bool) or value is None:
        raise ReportError("Mês inválido na planilha.")
    if isinstance(value, (int, float)):
        try:
            return (date(1899, 12, 30) + timedelta(days=int(value))).strftime("%Y-%m")
        except (OverflowError, ValueError):
            raise ReportError("Mês inválido na planilha.") from None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4}-\d{2})(?:-\d{2})?", text)
    if match:
        return match.group(1)
    for pattern in ("%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, pattern).strftime("%Y-%m")
        except ValueError:
            pass
    raise ReportError("Mês inválido na planilha.")


def normalized_url(value):
    parsed = urlparse(str(value or "").strip())
    host = (parsed.hostname or "").lower()
    if host not in ("instagram.com", "www.instagram.com", "facebook.com", "www.facebook.com"):
        raise ReportError("Link de criativo inválido na reconciliação.")
    return host.removeprefix("www.") + parsed.path.rstrip("/")


def _assert_close(label, actual, expected, tolerance=Decimal(0)):
    if abs(actual - expected) > tolerance:
        raise ReportError(f"Reconciliação divergente em {label}; nenhuma escrita realizada.")


def _candidate_metrics(row, expected):
    if expected == 0:
        return set()
    candidates = set()
    for action in row.get("actions", []):
        if action.get("action_type") and numeric(action.get("value")) == expected:
            candidates.add("action:" + action["action_type"])
    if "instagram_profile_visits" in row and numeric(row["instagram_profile_visits"]) == expected:
        candidates.add("field:instagram_profile_visits")
    return candidates


def _action_relation_counts(rows):
    """Describe candidate actions without exposing report values."""
    relevant = ("lead", "messag", "conversation", "profile", "contact")
    action_types = sorted({
        action.get("action_type", "")
        for row, _ in rows
        for action in row.get("actions", [])
        if action.get("action_type") and any(
            term in action.get("action_type", "").lower() for term in relevant
        )
    })
    summaries = []
    for action_type in action_types:
        counts = {"igual": 0, "menor": 0, "maior": 0, "ausente": 0}
        for row, expected in rows:
            entries = [
                action for action in row.get("actions", [])
                if action.get("action_type") == action_type
            ]
            if not entries:
                counts["igual" if expected == 0 else "ausente"] += 1
                continue
            if len(entries) > 1:
                raise ReportError("Evento duplicado na resposta; diagnóstico interrompido.")
            actual = numeric(entries[0].get("value"))
            counts["igual" if actual == expected else "menor" if actual < expected else "maior"] += 1
        signature = ",".join(f"{label}={counts[label]}" for label in ("igual", "menor", "maior", "ausente"))
        summaries.append(action_type + "[" + signature + "]")
    return summaries


def reconcile_sheet(raw, existing):
    """Compare Meta with the manually prepared closed month, without exposing row data."""
    if not existing or list(existing[0][:16]) != HEADERS:
        raise ReportError("Cabeçalhos da planilha mudaram; nenhuma escrita realizada.")
    month_rows = [(row + [""] * 16)[:16] for row in existing[1:] if row and sheet_month(row[0]) == raw["month"]]
    if not month_rows:
        raise ReportError("O mês de validação não existe na planilha.")
    sheet_rows = [
        {"platform": str(row[1]), "url": normalized_url(row[4]), "row": row}
        for row in month_rows
    ]

    used = set()
    matched_by_url = 0
    matched_by_metrics = 0
    classifications = {}
    result_sets = defaultdict(list)
    visits_sets = []
    missing_result_actions = defaultdict(set)
    missing_visit_actions = set()
    result_diagnostic_rows = defaultdict(list)
    for meta_row in raw["rows"]:
        platform = PLATFORMS.get(meta_row.get("publisher_platform"))
        creative = raw["ads"].get(str(meta_row.get("ad_id")), {}).get("creative", {})
        meta_url = normalized_url(creative.get("instagram_permalink_url"))
        url_candidates = [
            index for index, candidate in enumerate(sheet_rows)
            if index not in used and candidate["platform"] == platform
            and candidate["url"] == meta_url
        ]
        sheet_index = url_candidates[0] if len(url_candidates) == 1 else None
        if sheet_index is not None:
            matched_by_url += 1
        else:
            candidates = []
            for candidate_index, candidate in enumerate(sheet_rows):
                if candidate_index in used or candidate["platform"] != platform:
                    continue
                candidate_row = candidate["row"]
                same_counts = (
                    numeric(meta_row["reach"]) == sheet_number(candidate_row[6])
                    and numeric(meta_row["impressions"]) == sheet_number(candidate_row[7])
                    and numeric(meta_row.get("inline_link_clicks", 0)) == sheet_number(candidate_row[10])
                )
                same_spend = abs(numeric(meta_row["spend"]) - sheet_number(candidate_row[5])) <= Decimal("0.02")
                if same_counts and same_spend:
                    candidates.append(candidate_index)
            if len(candidates) != 1:
                raise ReportError(
                    "Pareamento incompleto entre Meta e planilha "
                    f"(Meta: {len(raw['rows'])}; planilha: {len(sheet_rows)}; "
                    f"por URL: {matched_by_url}; por métricas: {matched_by_metrics})."
                )
            sheet_index = candidates[0]
            matched_by_metrics += 1
        used.add(sheet_index)
        sheet_row = sheet_rows[sheet_index]["row"]
        _assert_close("investimento", numeric(meta_row["spend"]), sheet_number(sheet_row[5]), Decimal("0.02"))
        _assert_close("alcance", numeric(meta_row["reach"]), sheet_number(sheet_row[6]))
        _assert_close("impressões", numeric(meta_row["impressions"]), sheet_number(sheet_row[7]))
        _assert_close("cliques no link", numeric(meta_row.get("inline_link_clicks", 0)), sheet_number(sheet_row[10]))
        impressions = numeric(meta_row["impressions"])
        clicks = numeric(meta_row.get("inline_link_clicks", 0))
        spend = numeric(meta_row["spend"])
        _assert_close("CTR", clicks / impressions if impressions else Decimal(0), sheet_number(sheet_row[11]), Decimal("0.0001"))
        _assert_close("CPM", spend * 1000 / impressions if impressions else Decimal(0), sheet_number(sheet_row[12]), Decimal("0.02"))

        objective = str(sheet_row[3])
        if objective not in DESTINATIONS or str(sheet_row[2]) != DESTINATIONS[objective]:
            raise ReportError("Objetivo ou destino não corresponde ao padrão do dashboard.")
        adset = raw["adsets"].get(str(meta_row.get("adset_id")), {})
        promoted_fields = tuple(sorted(str(key) for key in adset.get("promoted_object", {})))
        classification_key = (
            str(adset.get("destination_type", "")),
            str(adset.get("optimization_goal", "")),
            promoted_fields,
        )
        previous = classifications.setdefault(classification_key, objective)
        if previous != objective:
            raise ReportError(
                "A classificação automática de objetivo ficou ambígua para "
                f"destination_type={classification_key[0]}, "
                f"optimization_goal={classification_key[1]}, "
                f"promoted_object_fields={','.join(classification_key[2]) or 'nenhum'}: "
                + ",".join(sorted((previous, objective)))
            )

        expected_result = sheet_number(sheet_row[8])
        result_diagnostic_rows[objective].append((meta_row, expected_result))
        candidates = _candidate_metrics(meta_row, expected_result)
        if expected_result and not candidates:
            missing_result_actions[objective].update(
                action.get("action_type", "") for action in meta_row.get("actions", [])
                if any(term in action.get("action_type", "").lower()
                       for term in ("lead", "messag", "conversation", "profile", "contact"))
            )
        if candidates:
            result_sets[objective].append(candidates)
        expected_visits = sheet_number(sheet_row[13])
        visit_candidates = _candidate_metrics(meta_row, expected_visits)
        if expected_visits and not visit_candidates:
            missing_visit_actions.update(
                action.get("action_type", "") for action in meta_row.get("actions", [])
                if "profile" in action.get("action_type", "").lower()
            )
        if visit_candidates:
            visits_sets.append(visit_candidates)

    if used != set(range(len(sheet_rows))):
        raise ReportError("Existem linhas na planilha sem anúncio correspondente na Meta.")
    total_sheet_spend = sum((sheet_number(row[5]) for row in month_rows), Decimal(0))
    total_meta_spend = sum((numeric(row["spend"]) for row in raw["totals"]), Decimal(0))
    _assert_close("total investido", total_meta_spend, total_sheet_spend, Decimal("0.02"))

    if missing_result_actions:
        summary = "; ".join(
            objective + ": " + ";".join(_action_relation_counts(result_diagnostic_rows[objective]))
            for objective in sorted(missing_result_actions)
        )
        raise ReportError("Resultados sem correspondência exata. Relações seguras: " + summary)
    if missing_visit_actions:
        raise ReportError("Visitas sem correspondência exata. Ações relevantes: " +
                          ",".join(sorted(filter(None, missing_visit_actions))))

    result_candidates = {}
    for objective, sets in result_sets.items():
        shared = set.intersection(*sets)
        if not shared:
            raise ReportError("A métrica Resultados não é consistente para um objetivo.")
        result_candidates[objective] = sorted(shared)
    visit_candidates = sorted(set.intersection(*visits_sets)) if visits_sets else []
    if visits_sets and not visit_candidates:
        raise ReportError("A métrica Visitas ao perfil não é consistente.")
    return {
        "month": raw["month"],
        "matched_rows": len(used),
        "matched_by_url": matched_by_url,
        "matched_by_metrics": matched_by_metrics,
        "classification_rules": [
            {"destination_type": key[0], "optimization_goal": key[1],
             "promoted_object_fields": list(key[2]), "objetivo": value}
            for key, value in sorted(classifications.items())
        ],
        "result_metric_candidates": result_candidates,
        "profile_visits_metric_candidates": visit_candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mapping", default=None)
    parser.add_argument("--check-sheet", action="store_true")
    args = parser.parse_args()
    try:
        client = MetaClient(os.environ.get("META_ACCESS_TOKEN"), os.environ.get("META_API_VERSION", "v26.0"))
        raw = collect(client, args.month)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        # Local evidence only. Never upload this automatically to a public CI artifact.
        (output / "meta_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.mapping:
            mapping = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
            prepared = report_rows(raw, mapping)
            (output / "meta_preview.json").write_text(json.dumps({"headers": HEADERS[:14], "rows": prepared}, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.check_sheet:
            existing = load_sheet_rows(os.environ.get("GCP_SERVICE_ACCOUNT_JSON"))
            reconciliation = reconcile_sheet(raw, existing)
            (output / "reconciliation.json").write_text(json.dumps(reconciliation, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Reconciliação concluída sem escrita: " + json.dumps(reconciliation, ensure_ascii=False, sort_keys=True))
        else:
            print("Coleta concluída. Nenhuma alteração foi feita na planilha.")
    except ReportError as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
