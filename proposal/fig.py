# Builds fig_mechanism.pdf from /home/claude/phase2/numbers.json (paired per-horizon AUROC, pretrained minus random, sd over 3 seeds)
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
n = json.load(open("/home/claude/phase2/numbers.json"))["paired_vs_rand"]
def ms(k):
    d = np.array(n[k]["d_h_auroc"]); return d.mean(), d.std(ddof=1)
plt.rcParams.update({"font.family":"serif","font.serif":["TeX Gyre Termes","Times New Roman","DejaVu Serif"],
                     "font.size":8,"axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,
                     "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 1.62), gridspec_kw={"width_ratios":[1.25,1]})
conds = [("pre","pretrained (published)"),("preE_newP","pretrained enc. + fresh predictor"),
         ("randE_preP","random enc. + pretrained predictor"),("tr","pretrained + target var/cov")]
cols = ["#444444","#888888","#c9c9c9","#ffffff"]
x = np.arange(2)*0.9; w = 0.15
for i,(c,lab) in enumerate(conds):
    m = [ms(f"{ds}|{c}|256|") for ds in ("FD001","FD003")]
    a.bar(x+(i-1.5)*w, [v[0] for v in m], w, yerr=[v[1] for v in m], color=cols[i], edgecolor="black",
          linewidth=0.5, error_kw={"elinewidth":0.6,"capsize":1.5}, label=lab)
a.axhline(0, color="black", linewidth=0.6)
a.set_xticks(x); a.set_xticklabels(["FD001","FD003"])
a.set_ylabel(r"$\Delta$ AUROC vs. random")
a.set_title("(a) where the deficit lives (d=256)", fontsize=8)
a.set_ylim(-0.056, 0.006); a.set_yticks([0,-0.01,-0.02,-0.03]); a.legend(fontsize=6.2, frameon=False, loc="lower center", ncol=2, handlelength=1.0, borderaxespad=0.1, columnspacing=1.0)
widths = [64,128,256]; mk = {"FD001":"o","FD003":"s","MBA":"^"}
for j,ds in enumerate(("FD001","FD003","MBA")):
    m = [ms(f"{ds}|pre|{d}|") for d in widths]
    off = (j-1)*0.04
    b.errorbar(np.log2(widths)+off, [v[0] for v in m], yerr=[v[1] for v in m], marker=mk[ds], ms=3.2,
               color="black", mfc="white" if ds=="MBA" else "black", linewidth=0.7, elinewidth=0.6, capsize=1.5, label=ds)
b.axhline(0, color="black", linewidth=0.6, linestyle=":")
b.set_xticks(np.log2(widths)); b.set_xticklabels(["64\n(0.16M)","128\n(0.57M)","256\n(2.16M)"])
b.set_xlabel("width d (params)", labelpad=1)
b.set_title("(b) pretrained minus random vs. width", fontsize=8)
b.set_ylim(-0.085,0.07); b.legend(fontsize=6.2, frameon=False, loc="lower right", ncol=3, handlelength=1.2)
fig.subplots_adjust(left=0.085, right=0.99, bottom=0.29, top=0.87, wspace=0.22)
fig.savefig("fig_mechanism.pdf"); fig.savefig("fig_mechanism.png", dpi=200)
for ds in ("FD001","FD003"):
    for c,_ in conds: print(ds, c, [round(v,4) for v in ms(f"{ds}|{c}|256|")])
