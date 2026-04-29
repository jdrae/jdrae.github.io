source "https://rubygems.org"

gem "jekyll", "~> 4.3"

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.17"
  gem "jekyll-seo-tag", "~> 2.8"
  gem "jekyll-sitemap", "~> 1.4"
  gem "jekyll-archives", "~> 2.3"
end

gem "webrick", "~> 1.8"

# Stdlib gems removed in Ruby 3.4+ but still expected by Jekyll 4
gem "csv"
gem "base64"
gem "bigdecimal"
gem "logger"

platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end
