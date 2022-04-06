from django import forms

SECURITIES_CHOICES = (
    ("INFY", "Infosys"),
    ("DIS", "Walt Disney"),
    ("GOOGL", "Google"),
    ("TSLA", "Tesla"),
    ("NFLX", "Netflix"),
    ("GRMN", "Garmin"),
    ("MSFT", "Microsoft"),
    ("KO", "Coca-Cola"),
    ("JPM", "JPMorgan & Chase"),
    ("WMT", "Walmart"),
    ("UBER", "Uber"),
    ("FB", "Meta"),
    ("AMZN", "Amazon"),
    ("REGN", "Regeneron Pharmaceuticals"),
    ("ZM", "Zoom"),
    ("VZ", "Verizon"),
    ("HD", "Home Depot"),
    ("NOC", "Northrop Grumman"),
    ("MCD", "McDonald's"),
    ("V", "Visa"),
    ("AXP", "American Express"),
    ("IBM", "IBM"),
    ("TXN", "Texas Instruments")
)


class SecuritiesForm(forms.Form):
    securities_field = forms.MultipleChoiceField(choices=SECURITIES_CHOICES,
                                                 widget=forms.CheckboxSelectMultiple(
                                                     attrs={"class": "form-group"}),
                                                 label="")
