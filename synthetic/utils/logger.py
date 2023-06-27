class PrinterLogger(object) : 
    def __init__(self, logger) : 
        self.logger = logger 
    def print_and_log(self, text) :
        self.logger.info(text)
        print(text)
    def info(self, text) : 
        self.logger.info(text)

