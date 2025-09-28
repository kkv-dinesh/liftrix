# cost_analyzer.py

import pandas as pd
import os
import difflib
import re
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import datetime


class CostAnalyzer:
	def analyze_shipment_cost(self, shipment_id):
		"""
		Given a shipment ID, explain why its cost was higher than expected.
		Looks up the shipment, compares to similar shipments, and provides possible reasons.
		"""
		if self.df.empty:
			return "No data available."
		
		# Look up shipment by Shipment_ID
		shipment = self.df[self.df['Shipment_ID'] == shipment_id]
		if shipment.empty:
			return f"Shipment ID {shipment_id} not found. Available IDs range from 1 to {len(self.df)}."
		
		# Get the first (and should be only) matching row
		shipment = shipment.iloc[0]
		# Already handled by the new implementation above
		cost = shipment['Cost_USD']
		origin = shipment['Origin']
		destination = shipment['Destination']
		product = shipment['Product_Type']
		date = shipment['Shipment_Date']
		weight = shipment['Weight_kg']
		volume = shipment['Volume_m3']
		# Find similar shipments (same route and product)
		similar = self.df[(self.df['Origin'] == origin) & (self.df['Destination'] == destination) & (self.df['Product_Type'] == product)]
		avg_cost = similar['Cost_USD'].mean() if not similar.empty else None
		msg = f"Shipment {shipment_id} from {origin} to {destination} ({product}) on {date}:\n"
		msg += f"- Cost: ${cost:,.2f}\n"
		if avg_cost is not None:
			msg += f"- Average cost for similar shipments: ${avg_cost:,.2f}\n"
			if cost > avg_cost:
				diff = cost - avg_cost
				msg += f"- This shipment cost ${diff:,.2f} more than average.\n"
				# Possible reasons
				reasons = []
				if weight > similar['Weight_kg'].mean():
					reasons.append("higher weight")
				if volume > similar['Volume_m3'].mean():
					reasons.append("higher volume")
				# Check if date is unusual (weekend, holiday, etc.)
				try:
					dt = pd.to_datetime(date)
					if dt.weekday() >= 5:
						reasons.append("shipment on weekend")
				except Exception:
					pass
				if reasons:
					msg += "Possible reasons: " + ", ".join(reasons) + ".\n"
				else:
					msg += "No obvious reason found in the data.\n"
			else:
				msg += "- This shipment did not cost more than average.\n"
		else:
			msg += "- No similar shipments found for comparison.\n"
		return msg.strip()

	def general_summary(self, filters=None, kpis=None, return_df=False):
		"""
		Generate a summary for any combination of filters and KPIs.
		filters: dict with keys like 'origin', 'destination', 'product_type', 'date', 'date_range'
		kpis: list of KPIs to compute (e.g., 'total_shipments', 'avg_cost', 'min_cost', 'max_cost', 'total_weight', 'avg_weight', 'total_volume', 'avg_volume', 'most_common_route', 'most_common_product')
		"""
		if self.df.empty:
			return ("No data available.", None) if return_df else "No data available."
		df = self.df.copy()
		df['Shipment_Date'] = pd.to_datetime(df['Shipment_Date'], errors='coerce')
		# Apply filters
		if filters:
			if filters.get('origin'):
				df = df[df['Origin'].str.lower() == filters['origin'].lower()]
			if filters.get('destination'):
				df = df[df['Destination'].str.lower() == filters['destination'].lower()]
			if filters.get('product_type'):
				df = df[df['Product_Type'].str.lower() == filters['product_type'].lower()]
			if filters.get('date'):
				df = df[df['Shipment_Date'].dt.date == filters['date']]
			if filters.get('date_range'):
				start, end = filters['date_range']
				df = df[(df['Shipment_Date'].dt.date >= start) & (df['Shipment_Date'].dt.date <= end)]
		if df.empty:
			return ("No shipments found for the specified filters.", None) if return_df else "No shipments found for the specified filters."
		# Default KPIs if not specified
		if not kpis:
			kpis = ['total_shipments', 'avg_cost', 'min_cost', 'max_cost', 'total_weight', 'avg_weight', 'total_volume', 'avg_volume', 'most_common_route', 'most_common_product']
		results = {}
		if 'total_shipments' in kpis:
			results['total_shipments'] = len(df)
		if 'avg_cost' in kpis:
			results['avg_cost'] = df['Cost_USD'].mean()
		if 'min_cost' in kpis:
			results['min_cost'] = df['Cost_USD'].min()
		if 'max_cost' in kpis:
			results['max_cost'] = df['Cost_USD'].max()
		if 'total_weight' in kpis:
			results['total_weight'] = df['Weight_kg'].sum()
		if 'avg_weight' in kpis:
			results['avg_weight'] = df['Weight_kg'].mean()
		if 'total_volume' in kpis:
			results['total_volume'] = df['Volume_m3'].sum()
		if 'avg_volume' in kpis:
			results['avg_volume'] = df['Volume_m3'].mean()
		if 'most_common_route' in kpis:
			try:
				results['most_common_route'] = df.groupby(['Origin', 'Destination']).size().idxmax()
			except Exception:
				results['most_common_route'] = None
		if 'most_common_product' in kpis:
			try:
				results['most_common_product'] = df['Product_Type'].mode()[0]
			except Exception:
				results['most_common_product'] = None
		# For charts
		cost_by_route = df.groupby(['Origin', 'Destination'])['Cost_USD'].sum().reset_index()
		cost_by_product = df.groupby('Product_Type')['Cost_USD'].sum().reset_index()
		shipments_by_day = df.groupby(df['Shipment_Date'].dt.date).size().reset_index(name='Shipments')
		# Add more: top N expensive/cheap shipments, top N routes, etc.
		top_expensive = df.nlargest(5, 'Cost_USD')[['Origin', 'Destination', 'Cost_USD', 'Product_Type', 'Shipment_Date']]
		top_cheap = df.nsmallest(5, 'Cost_USD')[['Origin', 'Destination', 'Cost_USD', 'Product_Type', 'Shipment_Date']]
		results['top_expensive'] = top_expensive
		results['top_cheap'] = top_cheap
		summary = "Custom Summary:\n"
		for k, v in results.items():
			if k in ['most_common_route'] and v:
				summary += f"- {k.replace('_', ' ').title()}: {v[0]} to {v[1]}\n"
			elif k in ['top_expensive', 'top_cheap']:
				continue
			elif v is not None:
				summary += f"- {k.replace('_', ' ').title()}: {v}\n"
		if return_df:
			return summary, {
				'cost_by_route': cost_by_route,
				'cost_by_product': cost_by_product,
				'shipments_by_day': shipments_by_day,
				'top_expensive': top_expensive,
				'top_cheap': top_cheap
			}
		return summary

	def generate_nl_summary(self, summary, charts, llm_client=None):
		"""
		Use the LLM to generate a natural language summary from the computed data.
		"""
		if not llm_client:
			return summary
		# Compose a prompt
		prompt = f"""
		You are a logistics data analyst. Given the following summary and data, write a concise, insightful natural language summary for a logistics manager.\n\nSummary:\n{summary}\n\nKey Data:\n- Cost by Route: {charts['cost_by_route'].to_string(index=False)}\n- Cost by Product: {charts['cost_by_product'].to_string(index=False)}\n- Shipments by Day: {charts['shipments_by_day'].to_string(index=False)}\n- Top 5 Expensive Shipments: {charts['top_expensive'].to_string(index=False)}\n- Top 5 Cheapest Shipments: {charts['top_cheap'].to_string(index=False)}\n"""
		return llm_client.chat(prompt, system_prompt="You are a logistics data analyst. Respond with a clear, concise summary for a manager.")

	def summarize_last_week(self, return_df=False):
		if self.df.empty:
			return ("No data available.", None) if return_df else "No data available."
		# Get today's date and last week's date range
		today = datetime.date.today()
		last_sunday = today - datetime.timedelta(days=today.weekday() + 1)
		last_monday = last_sunday - datetime.timedelta(days=6)
		# Filter for last week's shipments
		df = self.df.copy()
		df['Shipment_Date'] = pd.to_datetime(df['Shipment_Date'], errors='coerce')
		mask = (df['Shipment_Date'].dt.date >= last_monday) & (df['Shipment_Date'].dt.date <= last_sunday)
		week_df = df[mask]
		if week_df.empty:
			return ("No shipments found for last week.", None) if return_df else "No shipments found for last week."
		total_shipments = len(week_df)
		avg_cost = week_df['Cost_USD'].mean()
		min_cost = week_df['Cost_USD'].min()
		max_cost = week_df['Cost_USD'].max()
		total_weight = week_df['Weight_kg'].sum()
		avg_weight = week_df['Weight_kg'].mean()
		total_volume = week_df['Volume_m3'].sum()
		avg_volume = week_df['Volume_m3'].mean()
		most_common_route = week_df.groupby(['Origin', 'Destination']).size().idxmax()
		most_common_product = week_df['Product_Type'].mode()[0] if not week_df['Product_Type'].mode().empty else 'N/A'
		# For charts: cost by route, cost by product, shipments by day
		cost_by_route = week_df.groupby(['Origin', 'Destination'])['Cost_USD'].sum().reset_index()
		cost_by_product = week_df.groupby('Product_Type')['Cost_USD'].sum().reset_index()
		shipments_by_day = week_df.groupby(week_df['Shipment_Date'].dt.date).size().reset_index(name='Shipments')
		summary = (
			f"Last week ({last_monday} to {last_sunday}):\n"
			f"- Total shipments: {total_shipments}\n"
			f"- Average cost: ${avg_cost:,.2f}\n"
			f"- Min cost: ${min_cost:,.2f}\n"
			f"- Max cost: ${max_cost:,.2f}\n"
			f"- Total weight: {total_weight:,.2f} kg\n"
			f"- Average weight: {avg_weight:,.2f} kg\n"
			f"- Total volume: {total_volume:,.2f} m³\n"
			f"- Average volume: {avg_volume:,.2f} m³\n"
			f"- Most common route: {most_common_route[0]} to {most_common_route[1]}\n"
			f"- Most shipped product: {most_common_product}"
		)
		if return_df:
			return summary, {
				'cost_by_route': cost_by_route,
				'cost_by_product': cost_by_product,
				'shipments_by_day': shipments_by_day
			}
		return summary
	def __init__(self, data_path=None):
		self.data_path = data_path or os.path.join(os.path.dirname(__file__), '../data/freight_data.csv')
		self.df = self._load_data()

	def _load_data(self):
		try:
			# Load data and add a Shipment_ID column based on index
			df = pd.read_csv(self.data_path)
			df['Shipment_ID'] = df.index + 1  # Start IDs from 1
			return df
		except Exception as e:
			print(f"[CostAnalyzer] Error loading data: {e}")
			return pd.DataFrame()

	def get_cost_by_route(self, origin, destination, date=None, date_range=None, product_type=None):
		if self.df.empty:
			return "No data available."

		# Clean and normalize input
		def clean_city(city):
			if not city or not isinstance(city, str):
				return ""
			return re.sub(r'[^a-zA-Z ]', '', city).strip().lower()

		origin_clean = clean_city(origin)
		destination_clean = clean_city(destination)

		# Handle missing/partial info
		if not origin_clean and not destination_clean:
			return "Please specify both origin and destination."
		if not origin_clean:
			return "Please specify the origin city."
		if not destination_clean:
			return "Please specify the destination city."

		# Handle same city
		if origin_clean == destination_clean:
			return "Origin and destination cannot be the same."

		# Fuzzy match for misspelled city names
		all_origins = self.df['Origin'].dropna().unique()
		all_destinations = self.df['Destination'].dropna().unique()
		best_origin = difflib.get_close_matches(origin_clean.title(), all_origins, n=1, cutoff=0.7)
		best_destination = difflib.get_close_matches(destination_clean.title(), all_destinations, n=1, cutoff=0.7)
		if not best_origin:
			return f"Origin city '{origin}' not found. Did you mean one of: {', '.join(all_origins)}?"
		if not best_destination:
			return f"Destination city '{destination}' not found. Did you mean one of: {', '.join(all_destinations)}?"
		best_origin = best_origin[0]
		best_destination = best_destination[0]

		# Filter by route
		matches = self.df[(self.df['Origin'] == best_origin) & (self.df['Destination'] == best_destination)]

		# Filter by product type if provided
		if product_type:
			matches = matches[matches['Product_Type'].str.lower() == product_type.lower()]

		# Filter by date or date range if provided
		if date:
			matches = matches[matches['Shipment_Date'] == date]
		elif date_range:
			# date_range: tuple (start_date, end_date)
			try:
				start, end = [date_parser.parse(d).date() for d in date_range]
				matches = matches[(pd.to_datetime(matches['Shipment_Date']).dt.date >= start) & (pd.to_datetime(matches['Shipment_Date']).dt.date <= end)]
			except Exception:
				return "Could not parse the date range. Please specify valid dates."

		if matches.empty:
			msg = f"No shipments found from {best_origin} to {best_destination}"
			if product_type:
				msg += f" for {product_type}"
			if date:
				msg += f" on {date}"
			elif date_range:
				msg += f" between {date_range[0]} and {date_range[1]}"
			return msg + "."

		# Handle malformed/missing data
		costs = matches['Cost_USD'].tolist()
		dates = matches['Shipment_Date'].tolist()
		products = matches['Product_Type'].tolist()
		response = f"Found {len(costs)} shipment(s) from {best_origin} to {best_destination}"
		if product_type:
			response += f" for {product_type}"
		if date:
			response += f" on {date}"
		elif date_range:
			response += f" between {date_range[0]} and {date_range[1]}"
		response += ":\n"
		for i, (cost, d, prod) in enumerate(zip(costs, dates, products)):
			try:
				cost_val = float(cost)
				response += f"- On {d} ({prod}): ${cost_val:,.2f}\n"
			except Exception:
				response += f"- On {d} ({prod}): [Malformed cost data]\n"
		return response.strip()

	def parse_natural_language(self, query):
		"""
		Advanced: Parse a natural language query for origin, destination, product type, and date/date range.
		Handles synonyms, abbreviations, multi-step, and vague queries.
		Returns: dict with keys: origin, destination, product_type, date, date_range
		"""
		# Synonym/abbreviation maps
		synonyms = {
			'origin': ['from', 'source', 'start', 'pickup'],
			'destination': ['to', 'dest', 'end', 'dropoff', 'deliver'],
			'product_type': ['product', 'goods', 'item', 'cargo', 'type'],
			'cost': ['cost', 'price', 'charge', 'rate', 'fee'],
			'carrier': ['carrier', 'provider', 'vendor', 'company'],
			'route': ['route', 'path', 'lane', 'corridor'],
		}
		# Lowercase and remove punctuation for easier matching
		q = re.sub(r'[^a-zA-Z0-9\s-]', '', query.lower())

		# Replace synonyms with canonical terms
		for canon, syns in synonyms.items():
			for s in syns:
				q = re.sub(rf'\b{s}\b', canon, q)

		# Find product type first (so we can remove it from the string)
		product_types = self.df['Product_Type'].dropna().unique()
		product_type = None
		for pt in product_types:
			if pt.lower() in q:
				product_type = pt
				q = q.replace(pt.lower(), '')
				break

		# Find date or date range (remove from string after extraction)
		date = None
		date_range = None
		# Specific date
		date_match = re.search(r'on (\d{4}-\d{2}-\d{2})', q)
		if date_match:
			date = date_match.group(1)
			q = q.replace(date_match.group(0), '')
		# Date range
		range_match = re.search(r'between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})', q)
		if range_match:
			date_range = (range_match.group(1), range_match.group(2))
			q = q.replace(range_match.group(0), '')
		# Month/year phrases
		month_match = re.search(r'in ([a-zA-Z]+) (\d{4})', q)
		if month_match:
			try:
				month = month_match.group(1)
				year = int(month_match.group(2))
				start = date_parser.parse(f"1 {month} {year}").date()
				end = (start + relativedelta(months=1)) - datetime.timedelta(days=1)
				date_range = (str(start), str(end))
				q = q.replace(month_match.group(0), '')
			except Exception:
				pass
		# Last/this/next month/year
		today = datetime.date.today()
		if 'last month' in q:
			start = (today.replace(day=1) - relativedelta(months=1))
			end = today.replace(day=1) - datetime.timedelta(days=1)
			date_range = (str(start), str(end))
			q = q.replace('last month', '')
		elif 'this month' in q:
			start = today.replace(day=1)
			end = (start + relativedelta(months=1)) - datetime.timedelta(days=1)
			date_range = (str(start), str(end))
			q = q.replace('this month', '')
		elif 'next month' in q:
			start = (today.replace(day=1) + relativedelta(months=1))
			end = (start + relativedelta(months=1)) - datetime.timedelta(days=1)
			date_range = (str(start), str(end))
			q = q.replace('next month', '')
		elif 'last year' in q:
			start = today.replace(month=1, day=1) - relativedelta(years=1)
			end = today.replace(month=1, day=1) - datetime.timedelta(days=1)
			date_range = (str(start), str(end))
			q = q.replace('last year', '')
		elif 'this year' in q:
			start = today.replace(month=1, day=1)
			end = today.replace(month=12, day=31)
			date_range = (str(start), str(end))
			q = q.replace('this year', '')
		elif 'next year' in q:
			start = today.replace(month=1, day=1) + relativedelta(years=1)
			end = today.replace(month=12, day=31) + relativedelta(years=1)
			date_range = (str(start), str(end))
			q = q.replace('next year', '')

		# Now find origin and destination (after removing product/date phrases)
		# Support: 'from X to Y', 'origin X destination Y', 'pickup X dropoff Y', etc.
		origin = None
		destination = None
		# Try multiple patterns for flexibility
		patterns = [
			r'origin ([a-zA-Z ]+) destination ([a-zA-Z ]+)',
			r'from ([a-zA-Z ]+) to ([a-zA-Z ]+)',
			r'pickup ([a-zA-Z ]+) dropoff ([a-zA-Z ]+)',
			r'start ([a-zA-Z ]+) end ([a-zA-Z ]+)',
		]
		for pat in patterns:
			m = re.search(pat, q)
			if m:
				origin = m.group(1).strip()
				destination = m.group(2).strip()
				break

		# Fallback: try to extract single cities if only one is present
		if not origin:
			m = re.search(r'origin ([a-zA-Z ]+)', q)
			if m:
				origin = m.group(1).strip()
		if not destination:
			m = re.search(r'destination ([a-zA-Z ]+)', q)
			if m:
				destination = m.group(1).strip()

		return {
			'origin': origin,
			'destination': destination,
			'product_type': product_type,
			'date': date,
			'date_range': date_range
		}

# Example usage:
# analyzer = CostAnalyzer()
# print(analyzer.get_cost_by_route('Genoa', 'Rome'))

# Example usage:
# analyzer = CostAnalyzer()
# print(analyzer.analyze_shipment_cost(2345))
