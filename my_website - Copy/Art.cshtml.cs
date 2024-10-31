using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace WebApplication1.Pages
{
    public class ArtModel : PageModel
    {
        private readonly ILogger<ArtModel> _logger;

        public ArtModel(ILogger<ArtModel> logger)
        {
            _logger = logger;
        }

        public void OnGet()
        {
        }
    }
}
