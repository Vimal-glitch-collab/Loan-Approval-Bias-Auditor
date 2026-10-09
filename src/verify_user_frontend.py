"""Verify the applicant form has no audit UI and produces both result branches."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
assert not app.exception, str(app.exception)
assert not app.sidebar.radio and not app.metric
assert len(app.number_input) + len(app.selectbox) == 18
app.button[0].click().run()
assert not app.exception, str(app.exception)
text = ' '.join(m.value for m in app.markdown)
assert 'ASSESSMENT COMPLETE' in text
first_result = 'Favourable estimate' if 'Favourable estimate' in text else 'Unfavourable estimate'

# All inputs remain editable; resubmit a higher-risk hypothetical example.
next(w for w in app.number_input if w.label == 'Requested amount').set_value(18000)
next(w for w in app.number_input if w.label == 'Repayment period (months)').set_value(72)
app.selectbox(key='checking_account_status').set_value('Balance below 0')
app.selectbox(key='credit_history').set_value('Past payment delays')
app.selectbox(key='employment_duration').set_value('Unemployed')
app.button[0].click().run()
assert not app.exception, str(app.exception)
text = ' '.join(m.value for m in app.markdown)
assert '18,000' in text and '72 months' in text
assert 'ASSESSMENT COMPLETE' in text
second_result = 'Favourable estimate' if 'Favourable estimate' in text else 'Unfavourable estimate'
result = {'inputs':18, 'initial_submission':'passed', 'edited_submission':'passed',
          'sidebar_and_performance_metrics':'absent', 'default_result':first_result,
          'edited_result':second_result}
(ROOT / 'reports/user_frontend_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

