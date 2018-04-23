using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using ML_GDP.Models;
using Accord.Statistics.Models.Regression.Linear;
using Microsoft.AspNetCore.Session;
using Microsoft.Extensions.Caching.Memory;

namespace ML_GDP.Controllers
{
    public class HomeController : Controller
    {
        private IMemoryCache _cache;

        public HomeController(IMemoryCache memoryCache)
        {
            _cache = memoryCache;
        }

        public IActionResult Index()
        {
            ViewData["A"] = "A";
            ViewData["B"] = "B";
            ViewData["C"] = "C";
            return View();
        }

        [HttpPost]
        public IActionResult Index(IEnumerable<LeadingIndicator> leadingIndicators)
        {
            var regression = PerformRegression(leadingIndicators.ToList());
            _cache.Set<MultipleLinearRegression>("regression", regression);
            ViewData["A"] = regression.Weights[0];
            ViewData["B"] = regression.Weights[1];
            ViewData["C"] = regression.Intercept;
            return View();
        }


        private MultipleLinearRegression PerformRegression(List<LeadingIndicator> indicators)
        {
            double[][] inputs =
            {
                new double[] { indicators[0].StockIndex, indicators[0].M2Level },
                new double[] { indicators[1].StockIndex, indicators[1].M2Level },
                new double[] { indicators[2].StockIndex, indicators[2].M2Level },
                new double[] { indicators[3].StockIndex, indicators[3].M2Level },
            };

            double[] outputs = {
                indicators[0].GdpOutput,
                indicators[1].GdpOutput,
                indicators[2].GdpOutput,
                indicators[3].GdpOutput,
            };

            // We will use Ordinary Least Squares to create a
            // linear regression model with an intercept term
            var ols = new OrdinaryLeastSquares()
            {
                UseIntercept = true
            };

            // Use Ordinary Least Squares to estimate a regression model
            MultipleLinearRegression regression = ols.Learn(inputs, outputs);
            return regression;       
        }

        [HttpPost]
        public JsonResult Predict(double StockIndex, double M2Level)
        {
            double[] input = { StockIndex, M2Level };
            var regression = (MultipleLinearRegression)_cache.Get("regression");
            return new JsonResult(regression.Transform(input));
        }

       

        // to do: 1) increase training data 2) increase endogenous variables 3) use different learning model

        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}