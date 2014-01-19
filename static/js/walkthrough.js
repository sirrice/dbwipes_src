var Walkthrough = function(opts) {
  if (!opts || opts === undefined) {
    ops = {x: 500, y: 500}
  }
	this.x = opts.x;
	this.y = opts.y;
	this.el = $("#walkthrough-popover");
	this.wc = $("#walkthrough-container");
	this.el.find('.close').click($.proxy(this.close,this));


	this.el.css({
		left: this.x,
		top: this.y
	}).show();

	this.el.find('input')
		.change($.proxy(function(){
			this.render();
		}, this));

	this.el.find("#sb_addbad")
		.off()
		.on('click', $.proxy(function(){
			global_state.bad_keys = global_state.highlighted_keys.map(_.identity);
      $("#sb_addbad").toggleClass("clicked");
			this.render();
		}, this))

	this.el.find("#sb_addgood")
		.off()
		.on('click', $.proxy(function(){
			global_state.good_keys = global_state.highlighted_keys.map(_.identity);
      $("#sb_addgood").toggleClass("clicked");
			this.render();
		}, this))


	this.render();
}

_.extend(Walkthrough.prototype, Backbone.Events, {

	close: function() {
		this.el.hide();
	},
	
	render: function() {
    this.el.show();
    return;
	}

})



