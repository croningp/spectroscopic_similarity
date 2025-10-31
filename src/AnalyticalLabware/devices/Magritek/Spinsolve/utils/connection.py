"""
Python module for connection with Spinsolve NMR software.
"""

import queue
import re
import socket
import threading

from AnalyticalLabware.core.logging.loggers import get_logger


class SpinsolveConnection:
    """Provides API for the socket connection to the Spinsolve NMR instrument"""

    def __init__(self, host=None, port=13000):
        """
        Args:
            HOST (str, optional): TCP/IP address of the local host
            PORT (int, optional): TCP/IP listening port for Spinsolve software, 13000 by default
                must be changed in the software if necessary
        """

        # Getting the localhost IP address if not provided by instantiation
        # refer to socket module manual for details
        try:
            curr_host = socket.gethostbyname(socket.getfqdn())
        except socket.gaierror:
            curr_host = socket.gethostbyname(socket.gethostname())

        # Connection parameters
        self.host = curr_host if host is None else host
        self.port = port

        # The buffer size is so big for the only large message sent by the instrument - whole list of
        # Protocol options. One day will be reduced with addition of non-blocking parser/connection
        # TODO
        self.BUFSIZE = 2**16

        # Connection object, thread, lock and disconnection request tag
        self._listener = None
        self._connection = None
        self._connection_close_requested = threading.Event()

        # Response queue for inter threading commincation
        self.response_queue = queue.Queue()

        self.logger = get_logger(f"{self}")

    def __str__(self) -> str:
        return type(self).__name__

    def open_connection(self):
        """Open a socket connection to the Spinsolve software"""

        if self._connection is not None:
            self.logger.warning(
                "You are trying to open connection that is already open"
            )
            return

        # Creating socket
        self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connection.settimeout(None)

        # Connecting and spawning listening thread
        try:
            self._connection.connect((self.host, self.port))
        except ConnectionRefusedError:
            self._connection = None  # Resetting the internal attribute
            raise ConnectionRefusedError(
                "Please run Spinsolve software and enable remote control!"
            )
        self.logger.debug("Connection at %s:%s is opened", self.host, self.port)
        self._listener = threading.Thread(
            target=self.connection_listener,
            name=f"{__name__}_listener",
            daemon=False,
        )
        self._listener.start()
        self.logger.debug("Connection created")

    def connection_listener(self):
        """Checks for the new data and output it into receive buffer"""

        self.logger.debug("Connection listener thread is starting")

        while True:
            try:
                # Receiving data
                chunk = self._connection.recv(self.BUFSIZE)

                self.logger.debug("New chunk %s", chunk.decode())

                # split the msg
                msgs: list[str] = []
                msgs = re.split(rb"(?=\<\?xml)", chunk)

                for msg in msgs:
                    if msg.strip():
                        self.response_queue.put(msg)
                        self.logger.debug(f"Message {msg} added to the response queue")
            except ConnectionAbortedError:
                self.logger.warning("Connection aborted")
                break
            except ConnectionResetError:
                self.logger.warning("Spinsolve app is closed")
                break
            except OSError:
                self.logger.warning("Connection error")
                break
        self.logger.debug("Exiting listening thread")

    def transmit(self, msg):
        """Sends the message to the socket

        Args:
            msg (bytes): encoded message to be sent to the instrument
        """

        self.logger.debug("Sending the message")
        # This is necessary due to a random bug in the Spinsolve software with
        # wrong order of the messages sent.
        # See details in AnalyticalLabware/issues/22
        while True:
            try:
                unprocessed = self.response_queue.get_nowait()
                self.logger.error(
                    "Unprocessed message obtained from the response queue, \
see below:\n%s",
                    unprocessed,
                )
            except queue.Empty:
                break
        self._connection.send(msg)
        self.logger.debug("Message sent")

    def receive(self):
        """Grabs the message from receive buffer"""

        self.logger.debug("Receiving the message from the responce queue")
        reply = self.response_queue.get()
        self.response_queue.task_done()
        self.logger.debug("Message obtained from the queue")

        return reply

    def close_connection(self):
        """Closes connection"""

        self.logger.debug("Socket connection closure requested")
        self._connection_close_requested.set()
        if self._connection is not None:
            self._connection.shutdown(socket.SHUT_RDWR)
            self._connection.close()
            self._connection = None  # To available subsequent calls to open_connection after connection was once closed
            self._connection_close_requested.clear()
            self.logger.debug("Socket connection closed")
        else:
            self.logger.warning("You are trying to close nonexistent connection")
        if self._listener is not None and self._listener.is_alive():
            self._listener.join(timeout=3)

    def _flush_the_queue(self):
        while True:
            try:
                data = self.response_queue.get_nowait()
                if data:
                    self.logger.warning(
                        "Response queue flushed, something inside %s", data
                    )
                self.response_queue.task_done()
            except queue.Empty:
                break

    def is_connection_open(self):
        """Checks if the connection to the instrument is still alive"""
        # TODO
        raise NotImplementedError
