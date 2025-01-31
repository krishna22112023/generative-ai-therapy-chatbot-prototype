from abc import ABC, abstractmethod

class ConvoState(ABC):
    """
    Abstract base class for states in the state management system.
    Each state must implement enter, execute, and exit methods.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def enter(self, *context):
        """
        Handle actions to be performed when entering this state.
        """
        pass

    @abstractmethod
    def execute(self, *context):
        """
        Code to execute while this state is active.
        """
        pass

    @abstractmethod
    def end(self, *context):
        """
        Handle actions to be performed when exiting this state.
        """
        pass

class AppState(ABC):
    def __init__(self, username):
        self.username = username
    @abstractmethod
    def preNS(self, context):
        """
        Handle actions to be performed when entering this state.
        """
        pass

    @abstractmethod
    def duringNS(self, context):
        """
        Code to execute while this state is active.
        """
        pass

    @abstractmethod
    def postNS(self, context):
        """
        Handle actions to be performed when exiting this state.
        """
        pass