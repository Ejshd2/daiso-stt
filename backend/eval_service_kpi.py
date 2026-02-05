# backend/eval_service_kpi.py
import argparse
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()

    # 1) 기본 공백/기호 정리
    s = s.replace("·", " ").replace("/", " ").replace("-", " ")
    s = re.sub(r"[^\w\s가-힣]", " ", s)   # 구두점 제거(.,!? 등)
    s = re.sub(r"\s+", " ", s).strip()

    # 2) 숫자+단위 정규화
    # 센치/센티/센치미터 -> cm
    s = re.sub(r"(\d+)\s*(센치|센티|센티미터|센치미터)", r"\1cm", s)

    # 미리/밀리리터 -> ml
    s = re.sub(r"(\d+)\s*(미리|밀리|밀리리터|미리리터)", r"\1ml", s)

    # 그램/지 -> g (선택)
    s = re.sub(r"(\d+)\s*(그램|그람|g)", r"\1g", s)

    # 키로/킬로그램 -> kg
    s = re.sub(r"(\d+)\s*(키로|킬로|킬로그램|kg)", r"\1kg", s)

    # 매/매입/장/개입 같은 건 "숫자+단위" 붙여쓰기 통일
    s = re.sub(r"(\d+)\s*(매입|매|장|개입|개)", r"\1\2", s)

    # 3) “단어 사이 공백 제거 버전”도 비교하고 싶으면(선택)
    # 여기서는 normalize_text는 유지하고, keyword_hit에서 양쪽 버전 비교로 해결할게.

    return s



def parse_keywords(cell):
    """
    CSV 키워드 셀 → 리스트
    - "a|b|c" 또는 "a,b,c" 허용
    - NaN/None/"nan"/"" 는 빈 리스트로 처리
    """
    if cell is None or pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return []
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def keyword_hit(text: str, kw: str) -> bool:
    t = normalize_text(text)
    k = normalize_text(kw)
    if k == "":
        return False

    # 1) 기본 포함
    if k in t:
        return True

    # 2) 공백 제거 포함(실리콘 용기 vs 실리콘용기)
    t2 = t.replace(" ", "")
    k2 = k.replace(" ", "")
    return k2 != "" and k2 in t2


def basename_any(path_str: str) -> str:
    """ \\ / 둘 다 지원해서 파일명만 추출 """
    if path_str is None or pd.isna(path_str):
        return ""
    s = str(path_str).replace("\\", "/")
    return s.split("/")[-1]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # JSONL에는 file_name, provider가 이미 있음 (샘플 확인)
            audio_path = obj.get("audio_path") or obj.get("path") or obj.get("file_name")
            file_name = obj.get("file_name") or basename_any(audio_path)

            stt_text = obj.get("stt_text") or obj.get("text") or obj.get("final_transcript") or obj.get("transcript")

            rows.append({
                "audio_path_jsonl": audio_path,
                "file_name": file_name,
                "provider": obj.get("provider"),
                "model": obj.get("model"),
                "stt_text": stt_text,
            })

    df = pd.DataFrame(rows)
    df = df[df["file_name"].notna() & (df["file_name"] != "")].copy()
    df = df[df["provider"].notna() & (df["provider"] != "")].copy()
    return df


def build_default_out_path(jsonl_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"kpi_results_{jsonl_path.stem}_{ts}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="batch_results_*.jsonl")
    ap.add_argument("--manifest", required=True, help="test_manifest_core_option.csv")
    ap.add_argument("--out", default=None, help="output csv (optional)")
    ap.add_argument("--provider", default=None, help="whisper or google (optional filter)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    manifest_path = Path(args.manifest)
    out_path = Path(args.out) if args.out else build_default_out_path(jsonl_path)

    stt_df = load_jsonl(jsonl_path)
    man_df = pd.read_csv(manifest_path)

    required = ["audio_path", "expected_text", "gt_keywords_core", "gt_keywords_option"]
    for c in required:
        if c not in man_df.columns:
            raise ValueError(f"manifest missing column: {c}")

    # ✅ manifest에 file_name 만들기
    man_df["file_name"] = man_df["audio_path"].apply(basename_any)

    # ✅ provider 필터가 있으면 JSONL에서 먼저 줄이기
    if args.provider:
        stt_df = stt_df[stt_df["provider"] == args.provider].copy()

    # ✅ 핵심: provider별 결과를 정확히 붙이려면 file_name + provider 기준이 가장 안전
    # manifest는 provider가 없으니 file_name으로 붙인 뒤 provider 컬럼이 남는다 (파일당 2행이 정상)
    df = pd.merge(man_df, stt_df, on="file_name", how="left")

    # ----- KPI 계산 -----
    matched_core_col = []
    matched_opt_col = []
    core_success_col = []
    opt_cov_col = []

    for _, r in df.iterrows():
        text = r.get("stt_text") or ""
        core_list = parse_keywords(r.get("gt_keywords_core"))
        opt_list = parse_keywords(r.get("gt_keywords_option"))

        matched_core = [k for k in core_list if keyword_hit(text, k)]
        matched_opt = [k for k in opt_list if keyword_hit(text, k)]

        # is_core_success = (len(core_list) > 0) and (len(matched_core) == len(core_list))
        is_core_success = (len(core_list) > 0) and (len(matched_core) >= 1)
        opt_cov = None if len(opt_list) == 0 else (len(matched_opt) / len(opt_list))

        matched_core_col.append("|".join(matched_core))
        matched_opt_col.append("|".join(matched_opt))
        core_success_col.append(is_core_success)
        opt_cov_col.append(opt_cov)

    df["matched_core_keywords"] = matched_core_col
    df["matched_option_keywords"] = matched_opt_col
    df["service_success_core"] = core_success_col
    df["option_coverage"] = opt_cov_col

    # ✅ provider별 KPI 출력 (중요!)
    if "provider" in df.columns and df["provider"].notna().any():
        for prov, g in df.groupby("provider"):
            total = len(g)
            success_rate = (g["service_success_core"].sum() / total) if total else 0.0
            opt_mean = g["option_coverage"].dropna().mean()
            opt_mean = float(opt_mean) if pd.notna(opt_mean) else 0.0
            print(f"[KPI:{prov}] Service Success(core all-hit): {success_rate*100:.2f}% ({g['service_success_core'].sum()}/{total})")
            print(f"[KPI:{prov}] Option Coverage(avg): {opt_mean:.3f}")
    else:
        total = len(df)
        success_rate = (df["service_success_core"].sum() / total) if total else 0.0
        opt_mean = df["option_coverage"].dropna().mean()
        opt_mean = float(opt_mean) if pd.notna(opt_mean) else 0.0
        print(f"[KPI] Service Success(core all-hit): {success_rate*100:.2f}% ({df['service_success_core'].sum()}/{total})")
        print(f"[KPI] Option Coverage(avg): {opt_mean:.3f}")

    # ✅ 팀 공유용: provider/model 포함해서 “whisper vs google” 한눈에 비교
    team_view = df[[
        "file_name",
        "provider",
        "model",
        "expected_text",
        "gt_keywords_core",
        "stt_text",
        "matched_core_keywords",
        "service_success_core",
        "gt_keywords_option",
        "matched_option_keywords",
        "option_coverage",
    ]].copy()

    team_view.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
