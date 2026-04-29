# Mirrors jekyll-archives for the ko_posts collection: generates one page per
# category at /ko/categories/<slug>/, rendered with the `category` layout.
# Only generates pages for categories that have at least one ko_posts entry.

module Jekyll
  class KoCategoryPage < Page
    def initialize(site, base, category, posts)
      @site = site
      @base = base
      slug = Utils.slugify(category, mode: "default")
      @dir = File.join("ko", "categories", slug)
      @name = "index.html"

      process(@name)
      read_yaml(File.join(base, "_layouts"), "category.html")

      data["title"] = category
      data["lang"] = "ko"
      data["posts"] = posts.sort_by { |p| p.data["date"] }.reverse
      data["permalink"] = "/ko/categories/#{slug}/"
    end
  end

  class KoCategoryGenerator < Generator
    safe true
    priority :low

    def generate(site)
      ko_posts = site.collections["ko_posts"]&.docs || []
      return if ko_posts.empty?

      grouped = Hash.new { |h, k| h[k] = [] }
      ko_posts.each do |post|
        Array(post.data["categories"]).each do |category|
          grouped[category.to_s] << post
        end
      end

      grouped.each do |category, posts|
        site.pages << KoCategoryPage.new(site, site.source, category, posts)
      end
    end
  end
end
