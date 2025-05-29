document.getElementById("recommend-form").addEventListener("submit", function(e) {
          const button = this.querySelector("button[type=submit]");
          button.textContent = "Yükleniyor...";
          button.disabled = true;
          
          setTimeout(() => {
              button.textContent = "Öner";
              button.disabled = false;
          }, 2000);
      });