# compliance_checker.py
import pandas as pd
import os

class ComplianceChecker:
    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), '../data/freight_data.csv')
        self.df = self._load_data()

    def _load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            df['Shipment_ID'] = df.index + 1  # Start IDs from 1
            return df
        except Exception as e:
            print(f"[ComplianceChecker] Error loading data: {e}")
            return pd.DataFrame()

    def check_shipment(self, shipment_id):
        if self.df.empty:
            return "No data available."
        
        shipment = self.df[self.df['Shipment_ID'] == shipment_id]
        if shipment.empty:
            return f"Shipment ID {shipment_id} not found. Available IDs range from 1 to {len(self.df)}."
        
        # Get the first (and should be only) matching row
        shipment = shipment.iloc[0]
        
        # Check various compliance rules
        issues = []
        
        if shipment['Cost_USD'] > 10000:
            issues.append("High cost shipment requires additional approval")
            
        # Add more compliance checks based on the data we have
        # Weight and volume checks
        if shipment['Weight_kg'] > 5000:
            issues.append("Exceeds weight limit")
            
        if shipment['Volume_m3'] > 50:
            issues.append("Exceeds volume limit")
            
        if issues:
            return f"Shipment {shipment_id} has compliance issues: {', '.join(issues)}"
        return f"Shipment {shipment_id} is compliant."
        if row.empty:
            return f"Shipment {shipment_id} not found."
        # Example compliance checks (customize as needed):
        issues = []
        # Cost threshold
        if 'cost_usd' in row.columns and row['cost_usd'].iloc[0] > 10000:
            issues.append("Cost exceeds $10,000 limit.")
        # Delivery status
        if 'delivery_status' in row.columns and str(row['delivery_status'].iloc[0]).lower() != 'delivered':
            issues.append("Shipment not delivered.")
        # Carrier info
        if 'carrier' in row.columns and str(row['carrier'].iloc[0]).lower() == 'unknown':
            issues.append("Carrier information missing.")
        # Compliance flag
        compliance = row.iloc[0].get('compliance_flag', None)
        if compliance == 'flagged':
            issues.append("Compliance flag present.")
        if issues:
            return f"Shipment {shipment_id} is NOT compliant:\n- " + "\n- ".join(issues)
        return f"Shipment {shipment_id} is compliant."

    def flag_noncompliant_shipments(self):
        if self.df.empty:
            return []
        noncompliant = []
        for _, row in self.df.iterrows():
            issues = []
            if 'cost_usd' in row and row['cost_usd'] > 10000:
                issues.append("Cost exceeds $10,000 limit.")
            if 'delivery_status' in row and str(row['delivery_status']).lower() != 'delivered':
                issues.append("Shipment not delivered.")
            if 'carrier' in row and str(row['carrier']).lower() == 'unknown':
                issues.append("Carrier information missing.")
            if 'compliance_flag' in row and row['compliance_flag'] == 'flagged':
                issues.append("Compliance flag present.")
            if issues:
                noncompliant.append({
                    'shipment_id': row['shipment_id'],
                    'issues': issues
                })
        return noncompliant

# Agentic workflow stub for rebooking a shipment
def rebook_shipment(shipment_id, new_carrier=None):
    # In a real system, this would trigger an API call or workflow
    msg = f"Rebooking shipment {shipment_id}"
    if new_carrier:
        msg += f" with carrier {new_carrier}"
    msg += ". (Workflow triggered.)"
    return msg

# Example usage:
# checker = ComplianceChecker()
# print(checker.check_shipment(2345))
