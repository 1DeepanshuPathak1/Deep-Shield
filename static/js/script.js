//LOADER
jQuery(window).on("load", function () {
   $(".loader").fadeOut(800);

});
$(()=> {
   $(".scroll").on('click', function (event) {
      event.preventDefault();
      $('html,body').animate({
         scrollTop: $(this.hash).offset().top
      }, 1000);
   });

   if ($(window).width() > 768) {
      $(".parallaxie").parallaxie({
         speed: 0.55,
         offset: 0,
      });
   }
   $('.main-btn').on('mouseenter' , function(){
      $(this).addClass('active').siblings().removeClass('active');
   }).on('mouseleave' , function(){
      $('.main-btn').removeClass('active');
      $('.selected').addClass('active');
   });

//Wow Initializing
   if ($(".wow").length && $(window).outerWidth() >= 567) {
      let wow = new WOW({
         boxClass: 'wow',
         animateClass: 'animated',
         offset: 0,
         mobile: false,
         live: true
      });
      wow.init();
   }
});
