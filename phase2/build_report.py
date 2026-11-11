import subprocess, json, datetime
subprocess.run(["python3", "make_tables.py"], stdout=subprocess.DEVNULL)
subprocess.run(["python3", "gpu_est.py"], stdout=subprocess.DEVNULL)
tables = open("tables.md").read()
head = open("report_prose_top_built.md").read()
tail = open("report_prose_bottom.md").read()
stamp = subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
open("REPORT.md", "w").write(head.replace("{{STAMP}}", stamp) + "\n## Tables (auto-generated from /home/claude/phase2/out/*.json by make_tables.py)\n" + tables + "\n" + tail)
n = json.load(open("numbers.json")); n["gpu_estimate"] = json.load(open("gpu_est.json")); n["generated"] = stamp
json.dump(n, open("numbers.json", "w"), indent=1)
print("built", stamp)
