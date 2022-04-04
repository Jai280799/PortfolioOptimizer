from django.shortcuts import render
from django.views.decorators.cache import never_cache

from .Controller import processStocks, getData
from .forms import SecuritiesForm


def home(request):
    if request.method == 'POST':
        form = SecuritiesForm(request.POST)
        if form.is_valid():
            print(form.cleaned_data.get("securities_field"))
    else:
        form = SecuritiesForm()

    return render(request, 'home.html', {'form': form})


@never_cache
def charts(request):
    if request.method == 'POST':
        form = SecuritiesForm(request.POST)
        if form.is_valid():
            data = processStocks(list(form.cleaned_data.get("securities_field")))
            firstTicker = data['tickerList'][0]
            return render(request, 'charts.html', {
                "tickerList": data["tickerList"],
                "finalizedList": data["finalizedList"],
                "currentStock": firstTicker,
                "currentPred": data["{}_pred".format(firstTicker)],
                "currentSent": data["{}_sent".format(firstTicker)],
                "optimalPortfolio": data["optimalPortfolio"],
                "optimalPortfolioReturn": data["optimalPortfolioReturn"],
                "optimalPortfolioVolatility": data["optimalPortfolioVolatility"]})

    if request.method == 'GET':
        ticker = request.GET.get('ticker', '')
        data = getData()
        return render(request, 'charts.html', {
            "tickerList": data["tickerList"],
            "finalizedList": data["finalizedList"],
            "currentStock": ticker,
            "currentPred": data["{}_pred".format(ticker)],
            "currentSent": data["{}_sent".format(ticker)],
            "optimalPortfolio": data["optimalPortfolio"],
            "optimalPortfolioReturn": data["optimalPortfolioReturn"],
            "optimalPortfolioVolatility": data["optimalPortfolioVolatility"]})

    return render(request, 'home.html', {'form': SecuritiesForm()})


def about(request):
    return render(request, 'about.html', {})
