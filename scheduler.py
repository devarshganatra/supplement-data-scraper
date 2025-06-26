"""
Automated Scheduler for Market Intelligence Platform
===================================================

Handles automated data collection, analysis, and reporting.
Runs daily/weekly scraping jobs and generates automated reports.
"""

import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from market_intelligence_platform import MarketIntelligencePlatform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

class MarketIntelligenceScheduler:
    def __init__(self, db_path='supplement_market.db'):
        self.db_path = db_path
        self.platform = MarketIntelligencePlatform(db_path)
        self.is_running = False
        
    def run_daily_scraping(self):
        """Run daily data collection"""
        logging.info("🔄 Starting daily scraping job...")
        
        try:
            start_time = time.time()
            
            # Run the scraper
            self.platform.run_scraper()
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            
            # Log performance
            logging.info(f"✅ Daily scraping completed in {int(total_time)} seconds")
            
            # Save performance metrics
            self.save_performance_metric('daily_scraping_time', total_time)
            self.save_performance_metric('daily_scraping_success', 1)
            
        except Exception as e:
            logging.error(f"❌ Daily scraping failed: {str(e)}")
            self.save_performance_metric('daily_scraping_success', 0)
    
    def run_weekly_analysis(self):
        """Run weekly comprehensive analysis"""
        logging.info("📊 Starting weekly analysis job...")
        
        try:
            # Run market analysis
            category_analysis, market_gaps, price_trends = self.platform.run_market_analysis()
            
            # Build ML model
            ml_model = self.platform.build_ml_price_prediction_model()
            
            # Generate weekly report
            self.generate_weekly_report(category_analysis, market_gaps, price_trends, ml_model)
            
            logging.info("✅ Weekly analysis completed successfully")
            
        except Exception as e:
            logging.error(f"❌ Weekly analysis failed: {str(e)}")
    
    def run_monthly_cleanup(self):
        """Run monthly data cleanup and optimization"""
        logging.info("🧹 Starting monthly cleanup job...")
        
        try:
            # Clean old price history (keep last 6 months)
            cutoff_date = datetime.now() - timedelta(days=180)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old price history
            cursor.execute('''
                DELETE FROM price_history 
                WHERE recorded_at < ?
            ''', (cutoff_date.isoformat(),))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            logging.info(f"✅ Monthly cleanup completed. Deleted {deleted_rows} old records")
            
        except Exception as e:
            logging.error(f"❌ Monthly cleanup failed: {str(e)}")
    
    def generate_weekly_report(self, category_analysis, market_gaps, price_trends, ml_model):
        """Generate weekly analysis report"""
        try:
            report_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"weekly_report_{report_date}.txt"
            
            with open(filename, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("WEEKLY MARKET INTELLIGENCE REPORT\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Category Analysis
                f.write("📊 CATEGORY ANALYSIS\n")
                f.write("-" * 30 + "\n")
                for _, row in category_analysis.iterrows():
                    f.write(f"• {row['category']}: {row['total_products']} products, "
                           f"Avg price: €{row['avg_price']:.2f}\n")
                f.write("\n")
                
                # Market Gaps
                f.write("🔍 MARKET GAPS\n")
                f.write("-" * 30 + "\n")
                for category, gaps in market_gaps.items():
                    if gaps:
                        f.write(f"• {category}: {len(gaps)} gaps identified\n")
                f.write("\n")
                
                # Price Trends
                f.write("📈 PRICE TRENDS\n")
                f.write("-" * 30 + "\n")
                for category, trend in price_trends.items():
                    direction = "↗️ Increasing" if trend > 0 else "↘️ Decreasing" if trend < 0 else "➡️ Stable"
                    f.write(f"• {category}: {direction}\n")
                f.write("\n")
                
                # ML Model Performance
                if ml_model:
                    f.write("🤖 ML MODEL PERFORMANCE\n")
                    f.write("-" * 30 + "\n")
                    metrics = ml_model['metrics']
                    f.write(f"• Mean Absolute Error: €{metrics['mae']:.2f}\n")
                    f.write(f"• R² Score: {metrics['r2']:.3f}\n")
                f.write("\n")
                
                f.write("=" * 60 + "\n")
                f.write("End of Report\n")
                f.write("=" * 60 + "\n")
            
            logging.info(f"📄 Weekly report generated: {filename}")
            
        except Exception as e:
            logging.error(f"❌ Failed to generate weekly report: {str(e)}")
    
    def save_performance_metric(self, metric_name, metric_value):
        """Save performance metric to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (metric_name, metric_value)
                VALUES (?, ?)
            ''', (metric_name, metric_value))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"❌ Failed to save performance metric: {str(e)}")
    
    def start_scheduler(self):
        """Start the automated scheduler"""
        logging.info("🚀 Starting Market Intelligence Scheduler...")
        
        # Schedule jobs
        schedule.every().day.at("02:00").do(self.run_daily_scraping)
        schedule.every().sunday.at("03:00").do(self.run_weekly_analysis)
        schedule.every().month.at("04:00").do(self.run_monthly_cleanup)
        
        # Log scheduled jobs
        logging.info("📅 Scheduled Jobs:")
        logging.info("  • Daily scraping: 02:00 AM")
        logging.info("  • Weekly analysis: Sunday 03:00 AM")
        logging.info("  • Monthly cleanup: 1st of month 04:00 AM")
        
        self.is_running = True
        
        # Run scheduler in a separate thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logging.info("✅ Scheduler started successfully")
        
        # Keep main thread alive
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("🛑 Stopping scheduler...")
            self.stop_scheduler()
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        logging.info("✅ Scheduler stopped")
    
    def run_manual_job(self, job_type):
        """Run a manual job"""
        logging.info(f"🔧 Running manual {job_type} job...")
        
        if job_type == "scraping":
            self.run_daily_scraping()
        elif job_type == "analysis":
            self.run_weekly_analysis()
        elif job_type == "cleanup":
            self.run_monthly_cleanup()
        else:
            logging.error(f"❌ Unknown job type: {job_type}")

def main():
    """Main function to run the scheduler"""
    scheduler = MarketIntelligenceScheduler()
    
    # Check command line arguments for manual jobs
    import sys
    if len(sys.argv) > 1:
        job_type = sys.argv[1]
        scheduler.run_manual_job(job_type)
    else:
        # Start the automated scheduler
        scheduler.start_scheduler()

if __name__ == "__main__":
    main() 