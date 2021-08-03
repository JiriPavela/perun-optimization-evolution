from __future__ import annotations
from enum import Enum
from typing import Any


class OrderedEnum(Enum):
    """ An ordered enumeration structure that ranks the elements so that they can be compared in
    regards of their order. Taken from:
        https://stackoverflow.com/questions/42369749/use-definition-order-of-enum-as-natural-order

    :ivar int order: the order of the new element
    """

    def __init__(self, *args: Any) -> None:
        """ Create the new enumeration element and compute its order.

        :param args: additional element arguments
        """
        try:
            # attempt to initialize other parents in the hierarchy
            super().__init__(*args)
        except TypeError:
            # ignore -- there are no other parents
            pass
        ordered = len(self.__class__.__members__) + 1
        self.order = ordered

    def __ge__(self, other: OrderedEnum) -> bool:
        """ Comparison operator >=.

        :param other: the other enumeration element
        :return: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order >= other.order
        raise NotImplementedError

    def __gt__(self, other: OrderedEnum) -> bool:
        """ Comparison operator >.

        :param other: the other enumeration element
        :return: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order > other.order
        raise NotImplementedError

    def __le__(self, other: OrderedEnum) -> bool:
        """ Comparison operator <=.

        :param other: the other enumeration element
        :return: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order <= other.order
        raise NotImplementedError

    def __lt__(self, other: OrderedEnum) -> bool:
        """ Comparison operator <.

        :param other: the other enumeration element
        :return: the comparison result
        """
        if self.__class__ is other.__class__:
            return self.order < other.order
        raise NotImplementedError
