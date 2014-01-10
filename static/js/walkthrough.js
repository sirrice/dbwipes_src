var Walkthrough = function(opts) {
	this.x = opts.x;
	this.y = opts.y;
	this.el = $("#walkthrough-popover");
	this.wc = $("#walkthrough-container");
	this.next = $('#errbox_submit')
	this.el.find('.close').click($.proxy(this.close,this));
	this.prev = this.el.find('#errbox_prev');
	this.step = 0;	
	this.formstatus = new FormStatus();


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
			global_state.selected_keys = clone(global_state.highlighted_keys);
      $("#sb_addbad").toggleClass("clicked");
			//this.step += 1;
			this.render();
		}, this))
	this.el.find("#sb_addgood")
		.off()
		.on('click', $.proxy(function(){
			global_state.good_keys = clone(global_state.highlighted_keys);
      $("#sb_addgood").toggleClass("clicked");
			//this.step += 1;
			this.render();
		}, this))


	this.step = 0;
	this.render();
}

_.extend(Walkthrough.prototype, Backbone.Events, {

	close: function() {
		this.el.hide();
	},
	
	render: function() {
		var step = this.step;
		this.el.find('.block').hide();		
		var el = $("#walkthrough-" + step);
		el.show();



		if (step <= 0) {
			this.prev.attr('disabled', 'disabled');
		} else {
			this.prev
				.attr('disabled', null)
				.click($.proxy(function(event){
					event.stopPropagation();
					this.step -= 1;
					this.render();
					return false;
				}, this))			
		}


		if (step < 0) {
			this.next
				.val('next')
				.removeClass('btn-primary')
				.unbind()
				.click($.proxy(function(event){
					event.stopPropagation();
					this.step += 1;
					this.render();
					return false;
				}, this))
		} else {
			
			this.next
				.val('Submit')
				.addClass('btn-primary')
				.unbind()
				.click(function() {
					$("#overlay").show();
					$("#errbox").submit();
				});
		}


		this.trigger('change');
		this.formstatus.render();
		return this;
	}

})



var FormStatus = function(opts) {
	this.el = $("#formstatus-container");
	this.html = $("#formstatus-template").html();
//	this.template = Handlebars.compile(this.html);
	this.template = Handlebars.compile("{{#if nbad}}<div>{{nbad}}</div>{{/if}}");
}

_.extend(FormStatus.prototype, Backbone.Events, {

	count: function(keys) {
		var ret = 0;
		_.map(keys, function(v) {
				ret += v.length;
				return v.length
		})
		return ret;
	},

	render: function() {
            return;
		var nbad = this.count(global_state.selected_keys),
			ngood = this.count(global_state.good_keys),
			errtype = $("input[name=errtype]:checked").val(),
			errattrs = $("input.errattrs:checked");

		if (errtype != null && errtype != undefined) {
			if (errtype == 0) {
				errtype = "Wrong";
			} else if (errtype == 2) {
				errtype = "Too High"
			} else if (errtype == 3) {
				errtype = "Too Low"
			} else {
				errtype = null;
			}
		} else {
			errtype = null;
		}

		errattrs = _.map(errattrs, function(d) {
			return $(d).val();
		});

		nbad = (nbad == 0)? null : nbad+'';
		ngood = (ngood == 0)? null : ngood+'';

		var html = Handlebars.compile(this.html)({
				nbad: nbad,
				ngood: ngood,
				errtype: errtype,
				errattrs: errattrs
			});

		if (html.match(/^\s+$/)) {
			this.el.empty();
			return;
		}

		var newel = $($.parseHTML(html));
		this.el
			.empty()
			.append(newel)

		newel.find('#fs-reset-bad').click(function(){
			newel.find("#fs-bad").detach();
			global_state.selected_keys = {};
		});
		newel.find("#fs-reset-good").click(function() {
			newel.find("#fs-good").detach();
			global_state.good_keys = {};
		})
		newel.find("#fs-reset-errtype").click(function() {
			newel.find("#fs-errtype").detach();
			$("input[name=errtype]:checked").attr('checked', null);
		})

		return this;
	}
})




