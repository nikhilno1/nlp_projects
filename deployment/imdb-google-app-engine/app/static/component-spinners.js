(function ($) {
  $.fn.buttonLoader = function (action) {
    var self = $(this);
    var size_spinner = '';
    if (action == 'start') {
      if ($(self).attr('disabled') == 'disabled') {
        return false;
      }
      $(self).attr('data-btn-text', $(self).text());
      if($(self).attr('data-spinner-type')) {
        var type_spinner = $(self).attr('data-spinner-type');
      } else {
        var type_spinner = 'border';
      };
      if($(self).hasClass('btn-lg')) {
        var size_spinner = 'spinner-' + type_spinner + '-lg';
      };
      if($(self).hasClass('btn-sm')) {
        var size_spinner = 'spinner-' + type_spinner + '-sm';
      };
      if($(self).hasClass('btn-xs')) {
        var size_spinner = 'spinner-' + type_spinner + '-xs';
      };
      if($(self).attr('data-spinner-text')) {
        var text_spinner = '<span class="ml-2">' + $(self).attr('data-spinner-text') + '</span>';
      } else {
        var text_spinner = '<span class="sr-only d-block position-relative w-auto invisible" style="height:0;">' + $(self).attr('data-btn-text') + '</span>';
      };
      if($(self).hasClass('btn-spinner') != null) {
        $(self).html('<span class="spinner-' + type_spinner + ' ' + size_spinner + '" role="status" aria-hidden="true"></span>' + text_spinner );
      } else {
        $(self).html(text_spinner);
      };
      $(self).addClass('disabled');
    }
    if (action == 'stop') {
      $(self).html($(self).attr('data-btn-text'));
      $(self).attr('disabled', false).removeClass('disabled');
    }
  }
})(jQuery);
