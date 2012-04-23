#!/usr/bin/ruby
# -*- coding: utf-8 -*-

# https://github.com/mxcl/homebrew/issues/11816

HOMEBREW_BREW_FILE = ENV['HOMEBREW_BREW_FILE'] = File.expand_path("/usr/local/bin/brew")

require 'pathname'
HOMEBREW_LIBRARY_PATH = Pathname.new("/usr/local/bin/brew").realpath.dirname.parent.join("Library/Homebrew").to_s
$:.unshift(HOMEBREW_LIBRARY_PATH + '/vendor')
$:.unshift(HOMEBREW_LIBRARY_PATH)
require 'global'

    %w[CACHE CELLAR LIBRARY_PATH PREFIX REPOSITORY].each do |e|
      ENV["HOMEBREW_#{e}"] = Object.const_get "HOMEBREW_#{e}"
    end

  unlinked = HOMEBREW_CELLAR.children.reject do |rack|
    if not rack.directory?
      true
    elsif not (HOMEBREW_REPOSITORY/"Library/LinkedKegs"/rack.basename).directory?
      Formula.factory(rack.basename).keg_only? rescue nil
    else
      true
    end
  end.map{ |pn| pn.basename }


  unlinked.each do |l|
	puts "brew link #{l}"
	system "brew", "link", l
	exit 1 if $?.exitstatus != 0
  end


