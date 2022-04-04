from django import forms

SECURITIES_CHOICES = (
    ("DIS", "Walt Disney"),
    ("GOOGL", "Google"),
    ("TSLA", "Tesla"),
    ("NFLX", "Netflix"),
    ("MSFT", "Microsoft"),
    ("JPM", "JPMorgan & Chase"),
    ("WMT", "Walmart"),
    ("UBER", "Uber"),
    ("FB", "Meta"),
    ("AMZN", "Amazon"),
    ("ZM", "Zoom"),
    ("VZ", "Verizon"),
    ("HD", "Home Depot"),
    ("MCD", "McDonald's"),
    ("V", "Visa"),
    ("AXP", "American Express")
)


class SecuritiesForm(forms.Form):
    securities_field = forms.MultipleChoiceField(choices=SECURITIES_CHOICES,
                                                 widget=forms.CheckboxSelectMultiple(
                                                     attrs={"class": "form-group"}),
                                                 label="")
