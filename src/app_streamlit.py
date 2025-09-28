# Streamlit Demo App for Freight Optimization Assistant


import streamlit as st
import pandas as pd
from llm_client import LLMClient
from cost_analyzer import CostAnalyzer
from route_planner import RoutePlanner
from compliance_checker import ComplianceChecker
from rag_util import RAGUtil

st.set_page_config(page_title="Freight GenAI Assistant", layout="wide")
st.title("🚚 Freight GenAI Assistant")
st.write("""
Ask logistics questions and get real, data-driven answers powered by LLM and agent modules.
""")

sample_queries = [
	"Why did shipment 2345 cost $2,000 more than expected?",
	"Summarize last week's freight performance.",
	"Suggest the best carrier for route X.",
	"Rebook shipment 1234 with a different carrier.",
	"Is shipment 2345 compliant?"
]

# Initialize agents
llm = LLMClient()
cost_analyzer = CostAnalyzer()
route_planner = RoutePlanner()
compliance_checker = ComplianceChecker()
rag = RAGUtil()

# Define agentic_response function first
def agentic_response(query):
	# Use last 3 conversational turns as context for LLM
	memory = st.session_state["conversation_history"][-3:] if "conversation_history" in st.session_state else []
	# Use the advanced natural language parser for cost questions
	parsed = cost_analyzer.parse_natural_language(query)
	if parsed['origin'] and parsed['destination']:
		return cost_analyzer.get_cost_by_route(
			parsed['origin'],
			parsed['destination'],
			date=parsed['date'],
			date_range=parsed['date_range'],
			product_type=parsed['product_type']
		)
	q = query.lower()
	import re
	# Expanded: All possible optimal route/carrier queries
	route_keywords = [
		"optimal route", "best route", "suggest route", "which route", "route for", 
		"cheapest route", "most frequent route", "lowest cost route", "most reliable route", 
		"route from", "route to"
	]
	carrier_keywords = [
		"which carrier", "best carrier", "optimal carrier", "suggest carrier", 
		"cheapest carrier", "most reliable carrier", "carrier for", "carrier from", "carrier to"
	]
	if any(x in q for x in route_keywords + carrier_keywords):
		parsed_route = route_planner.parse_route_query(query)
		if any(x in q for x in carrier_keywords):
			return route_planner.suggest_optimal_carrier(**parsed_route)
		else:
			return route_planner.suggest_optimal_route(**parsed_route)
	if "cost" in q and "shipment" in q:
		match = re.search(r"shipment (\d+)", q)
		if match:
			shipment_id = int(match.group(1))
			return cost_analyzer.analyze_shipment_cost(shipment_id)
		return "Please specify a valid shipment ID."
	elif "summarize" in q:
		parsed = cost_analyzer.parse_natural_language(query)
		filters = {
			'origin': parsed['origin'],
			'destination': parsed['destination'],
			'product_type': parsed['product_type'],
			'date': None,
			'date_range': None
		}
		if parsed['date']:
			try:
				filters['date'] = pd.to_datetime(parsed['date']).date()
			except Exception:
				filters['date'] = None
		if parsed['date_range']:
			try:
				filters['date_range'] = tuple(pd.to_datetime(list(parsed['date_range'])).date)
			except Exception:
				filters['date_range'] = None
		summary, charts = cost_analyzer.general_summary(filters=filters, return_df=True)
		st.markdown(f"**Summary:**\n\n{summary}")
		if charts:
			nl_summary = cost_analyzer.generate_nl_summary(summary, charts, llm)
			st.markdown(f"**AI-Generated Summary:**\n\n{nl_summary}")
			st.markdown("---")
			st.subheader("Cost by Route")
			cost_by_route = charts['cost_by_route'].copy()
			cost_by_route['Route'] = cost_by_route['Origin'] + ' → ' + cost_by_route['Destination']
			st.bar_chart(cost_by_route.set_index('Route')['Cost_USD'])
			st.subheader("Cost by Product Type")
			st.bar_chart(charts['cost_by_product'].set_index('Product_Type')['Cost_USD'])
			st.subheader("Shipments by Day")
			shipments_by_day = charts['shipments_by_day'].copy()
			shipments_by_day = shipments_by_day.rename(columns={shipments_by_day.columns[0]: 'Date'})
			st.line_chart(shipments_by_day.set_index('Date')['Shipments'])
			st.subheader("Top 5 Expensive Shipments")
			st.dataframe(charts['top_expensive'])
			st.subheader("Top 5 Cheapest Shipments")
			st.dataframe(charts['top_cheap'])
		return ""
	elif ("compliant" in q or "compliance" in q or "flag" in q) and "shipment" in q:
		match = re.search(r"shipment (\d+)", q)
		if match:
			shipment_id = int(match.group(1))
			return compliance_checker.check_shipment(shipment_id)
		if "flag" in q or "all" in q:
			flagged = compliance_checker.flag_noncompliant_shipments()
			if not flagged:
				return "No noncompliant shipments found."
			return "Noncompliant shipments flagged:\n" + "\n".join([f"Shipment {x['shipment_id']}: {', '.join(x['issues'])}" for x in flagged])
		return "Please specify a valid shipment ID."
	elif ("rebook" in q or "rebooking" in q) and "shipment" in q:
		match = re.search(r"shipment (\d+)", q)
		if match:
			shipment_id = int(match.group(1))
			carrier_match = re.search(r"carrier ([\w\s]+)", q)
			new_carrier = carrier_match.group(1).strip() if carrier_match else None
			from compliance_checker import rebook_shipment
			return rebook_shipment(shipment_id, new_carrier)
		return "Please specify a valid shipment ID to rebook."
	else:
		# Compose context for LLM: last 3 user/assistant turns
		# Get conversation history context
		conv_context = ""
		for turn in memory:
			conv_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
		
		# Get relevant context from RAG
		relevant_chunks = rag.retrieve_relevant_context(query)
		rag_context = rag.format_context_for_llm(relevant_chunks)
		
		# Store chunks for debug display
		st.session_state.last_rag_chunks = relevant_chunks
		with debug_expander:
			st.write("Retrieved Context:")
			for chunk in relevant_chunks:
				st.markdown(f"""
				---
				**Similarity Score:** {chunk['similarity']:.3f}
				**Context:** {chunk['text']}
				**Metadata:** {chunk['metadata']}
				""")
		
		# Combine both types of context
		full_context = rag_context + "\n\nConversation history:\n" + conv_context + f"User: {query}\nAssistant:"
		
		return llm.chat(query, context=full_context)

# Create columns for main chat and sample queries
col1, col2 = st.columns([0.7, 0.3])

# Sample queries in the right column
with col2:
	st.markdown("### Sample Queries")
	for q in sample_queries:
		if st.button(q, key=q, use_container_width=True):
			# Set the query and add it to chat history immediately
			st.session_state["user_query"] = q
			# Process the query
			response = agentic_response(q)
			history = st.session_state["conversation_history"]
			history.append({"user": q, "assistant": response})
			if len(history) > 3:
				st.session_state["conversation_history"] = history[-3:]
			# Force a rerun to update the chat display
			st.rerun()


# --- Conversational Memory ---
if "conversation_history" not in st.session_state:
	st.session_state["conversation_history"] = []  # List of {user, assistant} dicts


# --- Chatbot UI: Chat history at top, input fixed at bottom ---
with col1:
	chat_placeholder = st.container()
	input_placeholder = st.empty()

# Add debug section in sidebar
debug_expander = st.sidebar.expander("🔍 Debug Info (RAG Context)", expanded=False)
if "last_rag_chunks" not in st.session_state:
    st.session_state.last_rag_chunks = []

with chat_placeholder:
	st.subheader("Chat History")
	for turn in st.session_state["conversation_history"]:
		st.markdown(f"<div style='margin-bottom:0.5em'><b>User:</b> {turn['user']}</div>", unsafe_allow_html=True)
		st.markdown(f"<div style='margin-bottom:1.5em'><b>Assistant:</b> {turn['assistant']}</div>", unsafe_allow_html=True)

# Input box always at the bottom
with input_placeholder:
	user_query = st.text_input("Type your logistics question here:", value=st.session_state.get("user_query", ""), key="chat_input")


if user_query:
	st.markdown("**Assistant Response:**")
	with st.spinner("Thinking..."):
		response = agentic_response(user_query)
	st.success(response)
	# Update conversational memory
	history = st.session_state["conversation_history"]
	# Get last assistant response (strip markdown)
	history.append({"user": user_query, "assistant": response})
	# Keep only last 3
	if len(history) > 3:
		st.session_state["conversation_history"] = history[-3:]

st.markdown("---")
st.caption("Powered by LLM and agentic tools. Data-driven answers.")
