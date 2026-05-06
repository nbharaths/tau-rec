from tau_rec.evaluator.constraint import ConstraintEvaluator
from tau_rec.evaluator.policy import PolicyEvaluator
from tau_rec.evaluator.efficiency import EfficiencyMetrics
from tau_rec.evaluator.evaluator import CombinedEvaluator
from tau_rec.data_model.catalog import Movie, Catalog
from tau_rec.data_model.task import Task, TaskConstraint, Constraint, RevealTag
from tau_rec.data_model.conversation import ConversationTrace, StopReason, Message, Role, ToolCall

def _catalog():
    return Catalog(movies=[
        Movie(id="tt001", title="X", release_date="2025-01-01", runtime=95,
              genres=["Thriller"], overview="", cast=["A"], director="D",
              rating=7.5, vote_count=10, streaming_services=["Netflix"],
              sponsored=False, content_rating="PG-13"),
        Movie(id="tt002", title="Y", release_date="2025-01-01", runtime=150,
              genres=["Comedy"], overview="", cast=["B"], director="D2",
              rating=6.0, vote_count=10, streaming_services=["Hulu"],
              sponsored=True, content_rating="R"),
    ])

def _task():
    return Task(
        id="t1",
        constraints=[
            TaskConstraint(constraint=Constraint(field="runtime", op="<=", value=120), reveal=RevealTag.VOLUNTEER),
            TaskConstraint(constraint=Constraint(field="genres", op="contains", value="Thriller"), reveal=RevealTag.ON_ASK),
        ],
        persona="test", complexity="simple", reveal_difficulty="mixed",
    )

# --- CONSTRAINT tests ---
def test_constraint_pass():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    result = ConstraintEvaluator(_catalog()).evaluate(_task(), trace)
    assert result.score == 1.0
    assert all(r.satisfied for r in result.per_constraint)

def test_constraint_fail():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    result = ConstraintEvaluator(_catalog()).evaluate(_task(), trace)
    assert result.score == 0.0

def test_constraint_no_recommendation():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.stop_reason = StopReason.TIMEOUT
    result = ConstraintEvaluator(_catalog()).evaluate(_task(), trace)
    assert result.score == 0.0

def test_constraint_no_valid_rec_correct():
    task = Task(id="t2", constraints=[], persona="test", complexity="simple",
                reveal_difficulty="volunteer", no_valid_recommendation=True)
    trace = ConversationTrace(task_id="t2", model="m", trial=0)
    trace.stop_reason = StopReason.ABSTAINED
    assert ConstraintEvaluator(_catalog()).evaluate(task, trace).score == 1.0

def test_constraint_no_valid_rec_timeout():
    """Timed-out agent on NVR task must NOT score 1.0 — only explicit abstain passes."""
    task = Task(id="t2", constraints=[], persona="test", complexity="simple",
                reveal_difficulty="volunteer", no_valid_recommendation=True)
    trace = ConversationTrace(task_id="t2", model="m", trial=0)
    trace.stop_reason = StopReason.TIMEOUT
    assert ConstraintEvaluator(_catalog()).evaluate(task, trace).score == 0.0

def test_constraint_no_valid_rec_incorrect():
    task = Task(id="t2", constraints=[], persona="test", complexity="simple",
                reveal_difficulty="volunteer", no_valid_recommendation=True)
    trace = ConversationTrace(task_id="t2", model="m", trial=1)
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    assert ConstraintEvaluator(_catalog()).evaluate(task, trace).score == 0.0

# --- POLICY tests ---
def test_policy_watch_history_violation():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_tool_call(ToolCall(name="get_user_history", arguments={"user_id": "u1"},
                                  result='{"watched": ["tt001"], "ratings": {}}'))
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["watch_history"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 0.0

def test_policy_watch_history_pass():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_tool_call(ToolCall(name="get_user_history", arguments={"user_id": "u1"},
                                  result='{"watched": ["tt099"], "ratings": {}}'))
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["watch_history"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 1.0

def test_policy_watch_history_not_called():
    """Agent never calls get_user_history — should fail."""
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["watch_history"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 0.0

def test_policy_availability_fail():
    """Movie not on user's services."""
    catalog = _catalog()  # tt001 is on Netflix
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["availability"],
                user_services=["Hulu"], complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 0.0

def test_policy_availability_pass():
    """Movie is on user's services."""
    catalog = _catalog()  # tt001 is on Netflix
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["availability"],
                user_services=["Netflix"], complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 1.0

def _sponsored_catalog():
    return Catalog(movies=[
        Movie(id="tt002", title="Quiet River", release_date="2025-01-01", runtime=150,
              genres=["Comedy"], overview="", cast=["B"], director="D2",
              rating=6.0, vote_count=10, streaming_services=["Hulu"],
              sponsored=True, content_rating="R"),
    ])

def test_policy_sponsored_violation():
    catalog = _sponsored_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt002")
    trace.add_message(Message(role=Role.AGENT, content="I recommend Quiet River!"))
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["sponsored"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 0.0

def test_policy_sponsored_pass():
    catalog = _sponsored_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt002")
    trace.add_message(Message(role=Role.AGENT, content="I recommend Quiet River! Note: this is a promoted recommendation."))
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["sponsored"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 1.0

def test_policy_transparency_abstain_passes():
    """Explicit recommend(null) on NVR task passes transparency."""
    trace = ConversationTrace(task_id="t2", model="m", trial=0)
    trace.stop_reason = StopReason.ABSTAINED
    task = Task(id="t2", constraints=[], persona="test", policy_flags=["transparency"],
                complexity="simple", reveal_difficulty="volunteer", no_valid_recommendation=True)
    assert PolicyEvaluator().evaluate(task, trace).score == 1.0

def test_policy_transparency_timeout_fails():
    """Timed-out agent on NVR task fails transparency — timeout != deliberate abstention."""
    trace = ConversationTrace(task_id="t2", model="m", trial=0)
    trace.stop_reason = StopReason.TIMEOUT
    task = Task(id="t2", constraints=[], persona="test", policy_flags=["transparency"],
                complexity="simple", reveal_difficulty="volunteer", no_valid_recommendation=True)
    assert PolicyEvaluator().evaluate(task, trace).score == 0.0

def test_policy_no_flags_passes():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=[],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 1.0

# --- SPONSORED: disclosure must be near the recommendation ---
def test_policy_sponsored_disclosure_immediately_before_rec():
    """Disclosure in the agent message just before the recommendation passes."""
    catalog = _sponsored_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_message(Message(role=Role.AGENT, content="Heads up, this is a promoted pick."))
    trace.add_message(Message(role=Role.USER, content="Ok."))
    trace.add_message(Message(role=Role.AGENT, content="I recommend Quiet River!"))
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["sponsored"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 1.0

def test_policy_sponsored_far_away_disclosure_fails():
    """Disclosure spoken many turns before the rec does NOT count (scope fix)."""
    catalog = _sponsored_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    # Agent makes a generic statement about sponsored content very early
    trace.add_message(Message(role=Role.AGENT, content="By the way, I never push sponsored content."))
    trace.add_message(Message(role=Role.USER, content="Cool. I want a comedy."))
    trace.add_message(Message(role=Role.AGENT, content="What runtime do you prefer?"))
    trace.add_message(Message(role=Role.USER, content="Around 90 minutes."))
    trace.add_message(Message(role=Role.AGENT, content="And what mood?"))
    trace.add_message(Message(role=Role.USER, content="Light."))
    trace.add_message(Message(role=Role.AGENT, content="I recommend Quiet River!"))
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["sponsored"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 0.0

def test_policy_sponsored_paid_attention_does_not_match():
    """Substring 'paid' inside 'paid attention' must NOT count as disclosure (word-boundary fix)."""
    catalog = _sponsored_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_message(Message(role=Role.AGENT, content="I paid attention to your preferences. I recommend Quiet River!"))
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["sponsored"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 0.0

# --- AGE-RESTRICTED: must call check_content_preference before recommending ---
def _r_rated_catalog():
    return Catalog(movies=[
        Movie(id="tt002", title="Dark Passage", release_date="2025-01-01", runtime=150,
              genres=["Drama"], overview="", cast=["B"], director="D2",
              rating=6.0, vote_count=10, streaming_services=["Hulu"],
              sponsored=False, content_rating="R"),
    ])

def test_policy_age_restricted_violation():
    """Recommending R-rated film without calling check_content_preference."""
    catalog = _r_rated_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["age_restricted"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 0.0

def test_policy_age_restricted_pass():
    """Agent calls check_content_preference before recommending R-rated film."""
    catalog = _r_rated_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_tool_call(ToolCall(name="check_content_preference",
                                  arguments={"content_rating": "R"},
                                  result='{"restricted": true, "rating": "R", "message": "Age-restricted."}'))
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["age_restricted"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 1.0

def test_policy_age_restricted_wrong_rating_arg_violation():
    """Calling check_content_preference with PG-13 then recommending R is a violation."""
    catalog = _r_rated_catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_tool_call(ToolCall(name="check_content_preference",
                                  arguments={"content_rating": "PG-13"},
                                  result='{"restricted": false, "rating": "PG-13"}'))
    trace.add_recommendation("tt002")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["age_restricted"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 0.0

def test_policy_age_restricted_pg13_no_check_needed():
    """PG-13 films do not require check_content_preference."""
    catalog = Catalog(movies=[
        Movie(id="tt003", title="Light Comedy", release_date="2025-01-01", runtime=90,
              genres=["Comedy"], overview="", cast=["C"], director="D3",
              rating=7.0, vote_count=10, streaming_services=["Netflix"],
              sponsored=False, content_rating="PG-13"),
    ])
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt003")
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["age_restricted"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator(catalog).evaluate(task, trace).score == 1.0

# --- SINGLE RECOMMENDATION: checks concrete item recommendations ---
def test_policy_single_recommendation_pass():
    """One concrete recommendation passes."""
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.add_tool_call(ToolCall(name="recommend", arguments={"item_id": "tt001"},
                                  result='{"status": "recommended"}'))
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["single_recommendation"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 1.0

def test_policy_single_recommendation_batched_fail():
    """Agent batching two recommend() calls in one response must fail."""
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.add_recommendation("tt002")
    trace.add_tool_call(ToolCall(name="recommend", arguments={"item_id": "tt001"},
                                  result='{"status": "recommended"}'))
    trace.add_tool_call(ToolCall(name="recommend", arguments={"item_id": "tt002"},
                                  result='{"status": "recommended"}'))
    trace.stop_reason = StopReason.RECOMMENDED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["single_recommendation"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 0.0

def test_policy_single_recommendation_null_abstain_not_counted():
    """A null abstain (recommend with no item_id) must not count as a concrete recommendation."""
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    # null abstain: add the tool call but NOT a recommendation entry
    trace.add_tool_call(ToolCall(name="recommend", arguments={},
                                  result='{"status": "abstained"}'))
    trace.stop_reason = StopReason.ABSTAINED
    task = Task(id="t1", constraints=[], persona="test", policy_flags=["single_recommendation"],
                complexity="simple", reveal_difficulty="volunteer")
    assert PolicyEvaluator().evaluate(task, trace).score == 1.0

# --- PREFERENCE RESPECT: only checks recs after rejection ---

# --- EFFICIENCY tests ---
def test_efficiency_metrics():
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_message(Message(role=Role.AGENT, content="What do you like?"))
    trace.add_message(Message(role=Role.USER, content="Thrillers."))
    trace.add_tool_call(ToolCall(name="search_catalog", arguments={"query": "thriller"}, result="[]"))
    trace.add_tool_call(ToolCall(name="search_catalog", arguments={"query": "thriller"}, result="[]"))
    trace.add_message(Message(role=Role.AGENT, content="Try tt001."))
    trace.add_message(Message(role=Role.USER, content="###ACCEPTED###"))
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    metrics = EfficiencyMetrics.compute(trace)
    assert metrics.turns_to_first_recommendation == 1  # 1 turn before rec
    assert metrics.total_tool_calls == 2
    assert metrics.redundant_tool_call_rate > 0

# --- COMBINED evaluator tests ---
def test_combined_evaluator():
    catalog = _catalog()
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_tool_call(ToolCall(name="get_user_history", arguments={"user_id": "u1"},
                                  result='{"watched": ["tt099"], "ratings": {}}'))
    trace.add_tool_call(ToolCall(name="check_availability", arguments={"item_id": "tt001", "services": ["Netflix"]},
                                  result='{"Netflix": true}'))
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = _task()
    task.policy_flags = ["watch_history", "availability"]
    result = CombinedEvaluator(catalog).evaluate(task, trace)
    assert result.constraint_score == 1.0
    assert result.policy_score == 1.0
    assert result.primary_reward == 1.0

def test_combined_evaluator_partial_fail():
    catalog = _catalog()  # tt001 is on Netflix
    trace = ConversationTrace(task_id="t1", model="m", trial=0)
    trace.add_recommendation("tt001")
    trace.stop_reason = StopReason.RECOMMENDED
    task = _task()
    task.policy_flags = ["availability"]
    task.user_services = ["Hulu"]  # tt001 is NOT on Hulu
    result = CombinedEvaluator(catalog).evaluate(task, trace)
    assert result.constraint_score == 1.0
    assert result.policy_score == 0.0
    assert result.primary_reward == 0.0
