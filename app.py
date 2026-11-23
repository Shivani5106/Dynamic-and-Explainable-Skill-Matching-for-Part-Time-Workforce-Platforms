"""Dashboard prototype (presentation layer of Fig. 3).  Run:  python app.py  ->  http://127.0.0.1:5000
Shows ranked workers for a task with factor-level explanations, and lets a task provider submit
completed-task feedback, which updates the worker's profile (verification-and-feedback loop)."""
import json, os
import numpy as np
from flask import Flask, request, redirect, url_for, render_template_string
from skillmatch.simulate import Params
from skillmatch.pipeline import simulate
from skillmatch.taxonomy import SKILL_NAMES, CLUSTER_NAMES
from skillmatch.model import STATUS

app = Flask(__name__)
LAM = 0.3
wpath = os.path.join(os.path.dirname(__file__), "results", "weights.json")
WEIGHTS = np.array(json.load(open(wpath))["tuned"]["Full framework"]["weights"]) if os.path.exists(wpath) \
    else np.array([.2, .25, .1, .2, .1, .15])


def build_demo(seed=0, epochs=3):
    for ed in simulate(seed, Params(), lams=(LAM,)):
        if ed.epoch == epochs:
            return ed.world, ed.platforms[LAM], ed.tasks


WORLD, PLAT, TASKS = build_demo()

PAGE = """
<!doctype html><title>Skill-matching prototype</title>
<style>body{font-family:system-ui;margin:2rem;max-width:1100px}table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:6px 8px;font-size:14px;text-align:left}th{background:#f3f3f3}
.bad{color:#b00020}.ok{color:#1b6e2d}small{color:#555}form{display:inline}</style>
<h2>Task &rarr; worker recommendations</h2>
<form method="get" action="/"><label>Task #</label>
<select name="t" onchange="this.form.submit()">
{% for i in range(n_tasks) %}<option value="{{i}}" {{'selected' if i==t else ''}}>{{i}} - {{cats[i]}}</option>{% endfor %}
</select></form>
<p><b>Required skills:</b> {{req}} &nbsp; <b>Category:</b> {{cat}} &nbsp; <b>Pay:</b> {{pay}}
&nbsp; <b>Mandatory verified skill:</b> {{mand}}</p>
{% if msg %}<p class="ok">{{msg}}</p>{% endif %}
<table><tr><th>#</th><th>Worker</th><th>Score</th><th>Why (factor contributions)</th><th>Skill status</th><th>Feedback</th></tr>
{% for r in recs %}<tr><td>{{loop.index}}</td><td>W{{r.w}}</td><td>{{'%.3f'%r.score}}</td>
<td>{% for f in r.ex.factors %}{{f.name}}: <b>{{'%.3f'%f.contribution}}</b><br>{% endfor %}
<small>{{r.ex.text}}</small></td>
<td>{% for s,st in r.status %}{{s}}: {{st}}<br>{% endfor %}</td>
<td><form method="post" action="/feedback"><input type="hidden" name="t" value="{{t}}">
<input type="hidden" name="w" value="{{r.w}}">
<input name="q" type="number" min="0" max="1" step="0.05" value="0.8" style="width:60px">
<button>Submit rating</button></form></td></tr>{% endfor %}</table>
<p><small>Ratings in [0,1] update p_new = (1-&lambda;)p_old + &lambda;E for each required skill; recommendations
re-rank immediately. Data are synthetic.</small></p>
"""


@app.route("/")
def index():
    t = int(request.args.get("t", 0))
    recs = []
    for w, sc in PLAT.recommend(TASKS, t, WEIGHTS, k=5):
        ex = PLAT.explain(TASKS, t, w, WEIGHTS)
        req = np.where(TASKS.req[t])[0]
        recs.append(dict(w=w, score=sc, ex=ex,
                         status=[(SKILL_NAMES[s], STATUS[int(PLAT.status[w, s])]) for s in req]))
    mand = SKILL_NAMES[TASKS.mand[t]] if TASKS.mand[t] >= 0 else "none"
    return render_template_string(
        PAGE, t=t, n_tasks=len(TASKS), cats=[CLUSTER_NAMES[c] for c in TASKS.cat], recs=recs, mand=mand,
        req=", ".join(SKILL_NAMES[s] for s in np.where(TASKS.req[t])[0]),
        cat=CLUSTER_NAMES[TASKS.cat[t]], pay=f"{TASKS.pay[t]:.0f}", msg=request.args.get("msg"))


@app.route("/feedback", methods=["POST"])
def feedback():
    t, w, q = int(request.form["t"]), int(request.form["w"]), float(request.form["q"])
    PLAT.record_outcome(w, int(TASKS.cat[t]), np.where(TASKS.req[t])[0], q)
    return redirect(url_for("index", t=t, msg=f"Rating {q:.2f} recorded for W{w}; profile updated."))


if __name__ == "__main__":
    app.run(debug=False)
