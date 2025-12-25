from abc import ABC, abstractmethod

class Manager(ABC):
    """
    Abstract base class for managing different projects.
    """

    @abstractmethod
    def build(self, flags):
        """
        Build the project with the given flags.
        """
        pass

    @abstractmethod
    def clean(self):
        """
        Clean the project.
        """
        pass

    @abstractmethod
    def test(self):
        """
        Test the project.
        """
        pass
