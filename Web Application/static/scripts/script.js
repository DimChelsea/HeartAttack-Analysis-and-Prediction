//smooth scrolling for nav
const navLinks = document.querySelectorAll("nav li a");
navLinks.forEach((element) => {
  element.addEventListener("click", () => {
    const targetSection = document.getElementById(
      element.getAttribute("data-target")
    );
    navLinks.forEach((elem) => {
      elem.removeAttribute("id");
    });
    element.id = "currentTarget";
    targetSection.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});
//smooth scrolling for footer

const footerLinks = document.querySelectorAll("footer li a");
footerLinks.forEach((element) => {
  element.addEventListener("click", () => {
    const targetSection = document.getElementById(
      element.getAttribute("data-target")
    );
    targetSection.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});
// handle humbergerMenu

const hamburgerBtn = document.querySelector("#hamburgerMenu");
const smallNav = document.querySelector("#smallNav");
const open = () => {
  smallNav.classList.replace("close", "open");
};
const close = () => {
  smallNav.classList.replace("open", "close");
};
let smallNavOpen = false;
hamburgerBtn.addEventListener("click", () => {
  if (smallNavOpen) close();
  else if (!smallNavOpen) open();
  smallNavOpen = !smallNavOpen;
});
const boxShadowForm = false;
// form box shadow changes on select
const inputRanges = document.querySelectorAll("input");
const form = document.querySelector("#patientForm ");
const selectInps = document.querySelectorAll("select");

inputRanges.forEach((inp) => {
  inp.addEventListener("input", () => {
    if (!boxShadowForm) {
      form.style.boxShadow = "0 0 3px #ef4f51";
      boxShadowForm;
    }
  });
});
selectInps.forEach((inp) => {
  inp.addEventListener("change", () => {
    if (!boxShadowForm) {
      form.style.boxShadow = "0 0 3px #ef4f51";
      boxShadowForm;
    }
  });
});

function changeBorder() {
  document.getElementById("selected").style.borderColor = "#ef4f51";  
  
}
function changeBorder1() {
  document.getElementById("selected1").style.borderColor = "#ef4f51";  
  
}
function changeBorder2() {
  document.getElementById("selected2").style.borderColor = "#ef4f51";  
  
}
function changeBorder3() {
  document.getElementById('selected3').style.borderColor = '#ef4f51';
}

function changeBorder4() {
  document.getElementById('selected4').style.borderColor = '#ef4f51';
}

// popup
//
// form.addEventListener('submit',function(event){
//   event.preventDefault()
//   document.querySelector('.bg-modal').style.display = "flex";
//   document.getElementById("outputImage").focus();
// })
//
//
// document.querySelector('.popupclose').addEventListener("click", function() {
// 	document.querySelector('.bg-modal').style.display = "none";
// });
//
// const box = document.querySelector(".modal-contents")
// // Detect all clicks on the document
// document.addEventListener("click", function(event) {
//   // If user clicks inside the element, do nothing
//   if (event.target.closest(".box")) return
//   // If user clicks outside the element, hide it!
//   box.classList.add("js-is-hidden")
// })