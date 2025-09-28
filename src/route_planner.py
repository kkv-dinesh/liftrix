
# ...existing imports...

class RoutePlanner:
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), '../data/freight_data.csv')
        self.df = self._load_data()

    def what_if_scenario(self, origin=None, destination=None, product_type=None, carrier=None, change=None):
        """
        Simulate 'what if' scenarios, e.g., 'What if I switch all shipments to carrier X?'
        """
        if self.df.empty:
            return "No data available."
        df = self.df.copy()
        if origin:
            df = df[df['Origin'].str.lower() == origin.lower()]
        if destination:
            df = df[df['Destination'].str.lower() == destination.lower()]
        if product_type:
            df = df[df['Product_Type'].str.lower() == product_type.lower()]
        if carrier:
            # Simulate switching all to this carrier
            carrier_df = df[df['Carrier'].str.lower() == carrier.lower()]
            if carrier_df.empty:
                return f"No historical data for carrier {carrier}."
            avg_cost = carrier_df['Cost_USD'].mean()
            return f"If all shipments used carrier {carrier}, the average cost would be ${avg_cost:,.2f}."
        if change == 'cheapest':
            cheapest = df.groupby('Carrier')['Cost_USD'].mean().idxmin()
            avg_cost = df[df['Carrier'] == cheapest]['Cost_USD'].mean()
            return f"If all shipments used the cheapest carrier ({cheapest}), average cost would be ${avg_cost:,.2f}."
        return "Please specify a scenario to simulate."

    def why_not_recommend(self, origin=None, destination=None, product_type=None, carrier=None):
        """
        Explain why a route/carrier is not recommended.
        """
        if self.df.empty:
            return "No data available."
        df = self.df.copy()
        if origin:
            df = df[df['Origin'].str.lower() == origin.lower()]
        if destination:
            df = df[df['Destination'].str.lower() == destination.lower()]
        if product_type:
            df = df[df['Product_Type'].str.lower() == product_type.lower()]
        if carrier:
            carrier_df = df[df['Carrier'].str.lower() == carrier.lower()]
            if carrier_df.empty:
                return f"Carrier {carrier} has no historical data for this route."
            avg_cost = carrier_df['Cost_USD'].mean()
            best_carrier = df.groupby('Carrier')['Cost_USD'].mean().idxmin()
            best_cost = df[df['Carrier'] == best_carrier]['Cost_USD'].mean()
            if avg_cost > best_cost:
                return f"Carrier {carrier} is not optimal because its average cost (${avg_cost:,.2f}) is higher than the best carrier ({best_carrier}, ${best_cost:,.2f})."
            return f"Carrier {carrier} is not recommended due to other factors (e.g., reliability, frequency)."
        return "Please specify a carrier to explain."

import pandas as pd
import os
import difflib
import re

class RoutePlanner:
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), '../data/freight_data.csv')
        self.df = self._load_data()

    def _load_data(self):
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            print(f"[RoutePlanner] Error loading data: {e}")
            return pd.DataFrame()

    def parse_route_query(self, query):
        """
        Parse user query for optimal route/carrier suggestions.
        Returns: dict with keys: origin, destination, product_type, date, date_range
        """
        q = re.sub(r'[^a-zA-Z0-9\s-]', '', query.lower())
        # Find origin and destination
        city_pattern = r'from ([a-zA-Z ]+) to ([a-zA-Z ]+)'
        city_match = re.search(city_pattern, q)
        origin = city_match.group(1).strip() if city_match else None
        destination = city_match.group(2).strip() if city_match else None
        # Find product type
        product_types = self.df['Product_Type'].dropna().unique()
        product_type = None
        for pt in product_types:
            if pt.lower() in q:
                product_type = pt
                break
        # Find date or date range
        date = None
        date_range = None
        date_match = re.search(r'on (\d{4}-\d{2}-\d{2})', q)
        if date_match:
            date = date_match.group(1)
        range_match = re.search(r'between (\d{4}-\d{2}-\d{2}) and (\d{4}-\d{2}-\d{2})', q)
        if range_match:
            date_range = (range_match.group(1), range_match.group(2))
        return {
            'origin': origin,
            'destination': destination,
            'product_type': product_type,
            'date': date,
            'date_range': date_range
        }

    def suggest_optimal_route(self, origin=None, destination=None, product_type=None, date=None, date_range=None):
        if self.df.empty:
            return "No data available."
        df = self.df.copy()
        # Fuzzy match for city names
        def fuzzy_match(val, options):
            if not val:
                return None
            matches = difflib.get_close_matches(val.title(), options, n=1, cutoff=0.7)
            return matches[0] if matches else None
        if origin:
            all_origins = self.df['Origin'].dropna().unique()
            origin = fuzzy_match(origin, all_origins)
            if not origin:
                return f"Origin city '{origin}' not found."
            df = df[df['Origin'] == origin]
        if destination:
            all_destinations = self.df['Destination'].dropna().unique()
            destination = fuzzy_match(destination, all_destinations)
            if not destination:
                return f"Destination city '{destination}' not found."
            df = df[df['Destination'] == destination]
        if product_type:
            df = df[df['Product_Type'].str.lower() == product_type.lower()]
        if date:
            df = df[pd.to_datetime(df['Shipment_Date'], errors='coerce').dt.date == pd.to_datetime(date).date()]
        if date_range:
            start, end = pd.to_datetime(list(date_range)).date
            df = df[(pd.to_datetime(df['Shipment_Date'], errors='coerce').dt.date >= start) & (pd.to_datetime(df['Shipment_Date'], errors='coerce').dt.date <= end)]
        if df.empty:
            return "No historical data found for the specified criteria."
        # Suggest optimal route: lowest average cost, highest frequency, etc.
        route_stats = df.groupby(['Origin', 'Destination']).agg(
            avg_cost=('Cost_USD', 'mean'),
            count=('Cost_USD', 'size')
        ).reset_index()
        best_route = route_stats.sort_values(['avg_cost', 'count'], ascending=[True, False]).iloc[0]
        response = (
            f"Optimal route based on historical data:\n"
            f"- Route: {best_route['Origin']} → {best_route['Destination']}\n"
            f"- Average cost: ${best_route['avg_cost']:,.2f}\n"
            f"- Number of shipments: {int(best_route['count'])}"
        )
        return response

    def suggest_optimal_carrier(self, origin=None, destination=None, product_type=None, date=None, date_range=None):
        # If carrier column exists in data, suggest best carrier for the route
        if self.df.empty():
            return "No data available."
        if 'Carrier' not in self.df.columns:
            return "Carrier information not available in the data."
        df = self.df.copy()
        if origin:
            all_origins = self.df['Origin'].dropna().unique()
            origin = difflib.get_close_matches(origin.title(), all_origins, n=1, cutoff=0.7)
            if not origin:
                return f"Origin city '{origin}' not found."
            df = df[df['Origin'] == origin[0]]
        if destination:
            all_destinations = self.df['Destination'].dropna().unique()
            destination = difflib.get_close_matches(destination.title(), all_destinations, n=1, cutoff=0.7)
            if not destination:
                return f"Destination city '{destination}' not found."
            df = df[df['Destination'] == destination[0]]
        if product_type:
            df = df[df['Product_Type'].str.lower() == product_type.lower()]
        if date:
            df = df[pd.to_datetime(df['Shipment_Date'], errors='coerce').dt.date == pd.to_datetime(date).date()]
        if date_range:
            start, end = pd.to_datetime(list(date_range)).date
            df = df[(pd.to_datetime(df['Shipment_Date'], errors='coerce').dt.date >= start) & (pd.to_datetime(df['Shipment_Date'], errors='coerce').dt.date <= end)]
        if df.empty:
            return "No historical data found for the specified criteria."
        carrier_stats = df.groupby('Carrier').agg(
            avg_cost=('Cost_USD', 'mean'),
            count=('Cost_USD', 'size')
        ).reset_index()
        best_carrier = carrier_stats.sort_values(['avg_cost', 'count'], ascending=[True, False]).iloc[0]
        response = (
            f"Optimal carrier based on historical data:\n"
            f"- Carrier: {best_carrier['Carrier']}\n"
            f"- Average cost: ${best_carrier['avg_cost']:,.2f}\n"
            f"- Number of shipments: {int(best_carrier['count'])}"
        )
        return response
