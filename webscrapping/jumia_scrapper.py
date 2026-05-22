import time
import pandas as pd
from playwright.sync_api import sync_playwright
import re

BASE_URL = "https://www.jumia.com.ng"
CATEGORY_URL = "https://www.jumia.com.ng/catalog/?q=phones" # Replace with category url
MAX_PRODUCTS = 50
FILE_NAME = "jumia_reviews_phones.csv"
FIRST_WRITE = True

def close_cookie_popup(page):

    cookie_button = page.locator("button")

    if cookie_button.count() > 0:
        for i in range(cookie_button.count()):
            btn_text = cookie_button.nth(i).inner_text().lower()

            if "accept" in btn_text or "agree" in btn_text:
                cookie_button.nth(i).click()
                print("Cookie popup closed")
                break

def scrape_jumia():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False
        )

        page = browser.new_page()

        print("Opening category page...")
        page.goto(CATEGORY_URL, timeout=120000)

        time.sleep(5)

        # -----------------------------------
        # GET PRODUCT LINKS
        # -----------------------------------

        product_links = []

        products = page.locator("a.core")

        count = products.count()

        print(f"Found {count} products")

        for i in range(count):
            try:
                href = products.nth(i).get_attribute("href")

                if href:
                    full_link = BASE_URL + href

                    if full_link not in product_links:
                        product_links.append(full_link)

            except Exception as e:
                print("Error getting product link:", e)

        print(f"Collected {len(product_links)} product links")

        # -----------------------------------
        # VISIT EACH PRODUCT
        # -----------------------------------

        for index, link in enumerate(product_links[:MAX_PRODUCTS]):
            data = []

            try:
                print(f"\n[{index+1}/{len(product_links)}]")
                print("Opening:", link)

                product_page = browser.new_page()

                product_page.goto(link, timeout=120000)

                time.sleep(4)

                # -----------------------------------
                # PRODUCT DETAILS
                # -----------------------------------

                try:
                    product_name = product_page.locator("h1").inner_text()
                except:
                    product_name = ""

                try:
                    price_text = product_page.locator("span.-b.-ubpt").first.inner_text()
                    price = price_text.replace("₦", "").replace(",", "").strip()
                except:
                    price = ""
                    
                # -----------------------------------
                # COUNT RATINGS
                # -----------------------------------

                try:
                    ratings_text = product_page.locator(
                        "a.-plxs._more"
                    ).first.inner_text()

                    match = re.search(r"\d+", ratings_text)

                    total_ratings = int(match.group()) if match else 0

                    print(f"Ratings: {total_ratings}")

                except:
                    total_ratings = 0
                    
                # -----------------------------------
                # PRODUCT AVG RATING
                # -----------------------------------

                try:
                    rating_text = product_page.locator(
                        "div.-fs29.-yl5 > span.-b"
                    ).first.inner_text()

                    avg_rating = float(rating_text.strip())

                    print(f"Avg Rating: {avg_rating}")

                except:
                    avg_rating = 0.0

                # -----------------------------------
                # OPEN REVIEWS PAGE
                # -----------------------------------

                try:
                    review_button = product_page.locator(
                        "a.btn._def._ti.-mhm.-fsh0"
                    )

                    if review_button.count() > 0:

                        button = review_button.first

                        if "See All" in button.inner_text():
                            try:
                                close_cookie_popup(product_page)
                            except:
                                pass

                            button.scroll_into_view_if_needed()

                            button.click(timeout=10000)

                            product_page.wait_for_load_state("networkidle")

                            time.sleep(2)

                            print("Opened reviews page")

                except Exception as e:
                    print("Could not open reviews page:", e)


                # -----------------------------------
                # SCRAPE REVIEWS + PAGINATION
                # -----------------------------------

                while True:

                    reviews = product_page.locator(
                        "article.-pvm.-hr._bet"
                    )

                    review_total = reviews.count()

                    print(f"Found {review_total} reviews on page")

                    for r in range(review_total):

                        try:
                            review = reviews.nth(r)

                            # -----------------------------------
                            # RATING
                            # -----------------------------------

                            try:
                                stars_style = review.locator(
                                    "div.in"
                                ).get_attribute("style")

                                rating = ""

                                if stars_style:

                                    if "100%" in stars_style:
                                        rating = 5

                                    elif "80%" in stars_style:
                                        rating = 4

                                    elif "60%" in stars_style:
                                        rating = 3

                                    elif "40%" in stars_style:
                                        rating = 2

                                    elif "20%" in stars_style:
                                        rating = 1

                            except:
                                rating = ""

                            # -----------------------------------
                            # TITLE
                            # -----------------------------------

                            try:
                                title = review.locator(
                                    "h3"
                                ).inner_text()

                            except:
                                title = ""

                            # -----------------------------------
                            # REVIEW TEXT
                            # -----------------------------------

                            try:
                                review_text = review.locator(
                                    "p"
                                ).first.inner_text()

                            except:
                                review_text = ""

                            # -----------------------------------
                            # REVIEWER + DATE
                            # -----------------------------------

                            reviewer = ""
                            review_date = ""

                            try:
                                meta_text = review.locator(
                                    "div.-df.-j-bet.-i-ctr.-gy5"
                                ).inner_text()

                                parts = meta_text.split("by")

                                review_date = parts[0].strip()

                                if len(parts) > 1:
                                    reviewer_raw = parts[1].strip()
                                    
                                    reviewer = reviewer_raw.replace("Verified Purchase", "").strip()

                            except:
                                pass

                            # -----------------------------------
                            # SAVE DATA
                            # -----------------------------------

                            row = {
                                "product_name": product_name,
                                "price": price,
                                "total_ratings": total_ratings,
                                "avg_rating": avg_rating,
                                "rating": rating,
                                "review_title": title,
                                "review_text": review_text,
                                "reviewer": reviewer,
                                "review_date": review_date,
                                "product_link": link
                            }

                            df = pd.DataFrame([row])
                            
                            global FIRST_WRITE

                            if FIRST_WRITE:
                                df.to_csv(FILE_NAME, index=False, mode="w")
                                FIRST_WRITE = False
                            else:
                                df.to_csv(FILE_NAME, index=False, mode="a", header=False)

                            print(row)

                        except Exception as e:
                            print("Review extraction error:", e)

                    # -----------------------------------
                    # NEXT PAGE
                    # -----------------------------------

                    try:
                        next_button = product_page.locator(
                            'a[aria-label="Next Page"]'
                        )

                        if next_button.count() > 0:

                            print("Going to next review page...")

                            next_button.click()

                            product_page.wait_for_load_state(
                                "networkidle"
                            )

                            time.sleep(2)

                        else:
                            break

                    except:
                        break
                    
                product_page.close()

            except Exception as e:
                print("Product visit error:", e)

    # # -----------------------------------
    # # SAVE DATA
    # # -----------------------------------

    # df = pd.DataFrame(data)

    # df.to_csv("jumia_reviews.csv", index=False)

    print("\nDONE.")
    print("DONE. Scraping complete.")


if __name__ == "__main__":
    scrape_jumia()