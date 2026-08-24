"""Step 6: report -- one Markdown report + one summary.json for the whole run."""

import json
import os

STEP_NAME = "report"


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def run(ctx) -> dict:
    types = ctx.load("detect_type")
    assessment = ctx.load("assess")
    recs = ctx.load("recommend")
    validation = ctx.load("validate")

    lines = ["# data-trance report", ""]
    lines.append(f"Input: `{ctx.config.input}`  ")
    lines.append(f"Output: `{ctx.config.output_dir}/transformed.csv`")
    lines.append("")

    summary = {}

    for col, info in types.items():
        vtype = info["type"]
        lines.append(f"## `{col}`")
        lines.append(f"**Type:** {vtype} ({info['reasoning']})")
        lines.append("")

        if vtype in ("categorical", "ordinal"):
            rec = recs.get(col, {})
            lines.append(f"**Recommended encoding:** {rec.get('recommendation')} "
                          f"-- {rec.get('reason')}")
            lines.append("")
            summary[col] = {"type": vtype, "encoding": rec.get("recommendation")}
            continue

        a = assessment.get(col, {}).get("stats", {})
        if a:
            lines.append(f"- n = {a['n']}, mean = {_fmt(a['mean'])}, "
                          f"skew = {_fmt(a['skew'])}, "
                          f"excess kurtosis = {_fmt(a['kurtosis_excess'])}")
            lines.append(f"- normality ({a['normality_test']}): "
                          f"p = {_fmt(a['normality_p'], 4)}")
        before_plot = assessment.get(col, {}).get("plot")
        if before_plot:
            lines.append(f"\n![before]({os.path.relpath(before_plot, ctx.config.output_dir)})")

        rec = recs.get(col, {})
        chosen = rec.get("chosen_transform")
        lines.append("")
        lines.append(f"**Recommended transform:** {chosen} ({rec.get('reason')})")

        v = validation.get(col, {})
        if v.get("validated"):
            after = v["after"]
            lines.append(f"- after: skew = {_fmt(after['skew'])}, "
                          f"excess kurtosis = {_fmt(after['kurtosis_excess'])}, "
                          f"improved = {v['improved']}")
            after_plot = v.get("plot")
            if after_plot:
                lines.append(f"\n![after]({os.path.relpath(after_plot, ctx.config.output_dir)})")
        lines.append("")

        summary[col] = {
            "type": vtype,
            "chosen_transform": chosen,
            "reason": rec.get("reason"),
            "improved": v.get("improved"),
        }

    report_path = os.path.join(ctx.config.output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    summary_path = os.path.join(ctx.config.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    result = {"report_md": report_path, "summary_json": summary_path}
    ctx.save(STEP_NAME, result)
    return result
