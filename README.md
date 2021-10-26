A simple implementation of SPSA in Python with automatic learning rate tuning.

To try it yourself without downloading anything, you can use the [Try It Online](https://tio.run/##vVltb9s4Ev7uXzFIcYDcOIpf4jrOXQI4RRcp0iuKtIv7kM0ljEXbwkmiIEqJvIv97b1nSL1azja7wJ2BVBJfhsOZZ54ZsvE23aho8n2VqJBCkW7ID2OVpLRUekCx3ys@oyyMtyQ0RXHPjE23sR@ty9HvRRCIx0AO6FsW8@PnyFdRr/eGAqW1I5JEbPt0dEGrQInU7X1C609ZtEwxis6r6be3UexGnhl@N7CD7757ckU61sLpEX6rM2rOHpjG/IzqmbYpSM6sAMj/rCJZtt57cim2dd9IHk2qvlg9y8a8oTu1XXFeN07k0UnZukfYuOrrCBu9G9nOUIUySrOw2Tm3XX4qE8Eb02fkR0bmED/budwuA9noGPRqo56ZEQcHB@b5MYQbeBEji9SKvn75uqBnHx4uVycReRRIkUTsSawqKZFaptrtGRlfRCJCCX20@Tyqf@bbOqN65d@3jaRV6dZHyWJVnPqh/6v0XONl6TGIHla7kHhwK0F5V6Yf@akvAooV7zzTLO3bxtf0JIIM3ZqkB8N5ZkeJTLMkwpBKDKDQEvlxRZHa2TlkrP0nGQ0o3ciIBC2TzJMkNdQv@lcqg/hM85zAjyRpSFhu3I66bclLFaWJCjQLBo4l1IQ70Pokk7WMlpLEcqkSjyekyoxaqSBQz9ywUkmYBaKtv7ESHZ1jY/SW1onwfDj0vtT1B/pgJ54IYxmxItCBMEvSky/IgNnY0ED3L@kFlY7JGdFhFWpQcZklCWtYgbtPb99W8daS8a/C@ox0kpGnjUO6W0gz1v9xCycYnPkrhA7pjcoCNEvyVPbIaFMJbUTw1IZDEbathd@XbtqoZ1oJndp1E2sVzUthx6kfB1tjEXSCFB9hJrVqRK3b3c1wUIlSEgyqwK4bEa3h@JQQE@RlSSlyn6CKl/6svjKPQXwRQic1obj6f2od74njWCaIzUfLSRqsUAcH7yLMlhsAG94tBsJ/QF4Vg7xUifa/BE4vB23GORCZAOUqvNf@OtJdNfP/cZBggSJIyhTyYpCUeeQ1QdI1L29DLhMptA0WAaQsU2W8/4Cc89D0125UVCmstfZXKevQNhmkgCeJR1iqFlgludb0q9LPBRrjRD75KtOVl3XZU3s6kYbzCbuSAlMrC3Wpt8pt2LkItLIZjSFr2DcVKFYg34gx5qtFNDJvR@y@mGEMuXSjsvUm2FqS2YmvKhVKzlO7AVIk8z9Yy47YiTBjDh0HYLss5kpAta3wodoadp3KME61QSb4sgxoA97cPBo2Spvmw4wlcy4WYBRbSKVKkQq8ojy4MXm2XRv8URIvHexRiHweYhE2UVlrgEvSlqwXs3c3o73Guysn73flFDFnqtSimEh4dpkuynLqDX1oklBreTsS@QeWBeq41qwX2plYYrrWa82UhNr1V5ko7eSu3ohY9qvuz6bOq3eB0L2HTxh7a@l83tlSzW@QaCnOXW6Uv5TO7dFoQCMU1Z0lCi0Oz8mBkcBJXo66jA2Gf/HeZ64aQybem2ofo65@aZ97DFQUB6ZGnrJIqIgqSgRrNwLCnDVWPeTOWb8hdREj34K2tM8FbbPsYoiufIYwU5xxYlJhzG1ZrOA8mM3hip7GA9Tbs@EMBnFPRic7Rnze@FjJbt8WWNDsomgoRL2tevZWQG/Pi4G9PeVa3muPhEoFxj4WZW7FfFU4Ij6z2G5qL2AG5OFAJs9NLW3t92xOFRY6bAS/hk1NJg31ffrbeZNmjgv6aTjjRtoQYw/bzgZ7ghrpHzRq2@MNzoPRHkhYCmqkq50KxFoGPGEdZtw0oP/I7XkgwkdP2NNdy0VtQHMQmFPVjjo3huq6Bq6yj7sbF68ydoWdhtErVZjanNWBAf9vQfL734l@SZmP@Jufvx80If/FmsWYKIu9TuHTxXgZ9t2awm/VEH@OFwpOGMHGpZkgjZwfswRa1g2ueN4vhxuemxv/ud6sVkFmUk7udkKoe7zwW8cJAweMejYz7WmQ8l6PrxFQbuo0EXDIfShip31tgCzwfGZvLm6NbwfUuo7Y@OvNy/3mLFt/t4/jNwK7QhGJAhr1GsJxyZFQa4MCbqsyWqLbR5sUSJIippt/RybDG5uY2OX0FiNZ@Tjc2ISlzUxPaj@Rrz@47@TnBcpJ1tosYkpXvuM5QvJnmj3Es0HisFL1YWzSSauVqoWG2HdqC3vsKjZV/WuKCP2CkrXglrpQjLVllQrpH3LBaWO/9H@KWNPtEejlCNziuu6AJgM6uaOWvZ0hf4@G/doAps6Ed9hbj5yOJJdG5nUbwu6Jv@RaOEvpwcnhWI5zB8cpBwoesnb943H/oRLHcR4qnK9QOJTVOvZUr3dxcbELXOBMWC5n/af9AR/XoGQtlA3l3M7d@WQynI1O5gOau6fj4WgymxO/T8fT2clojBx46p4OZ3Okw9MBzdzJaDg9nc6K26jGb@qWb0ig7rvTOSh5zNzhjubj4Xg@Nln1ZHYynp7OJvw@ms1P341HdNdvxUIRkeytQ@OsMqKxKZxYHaZyE6xclN2bzGdv/j6Ziy@r2FXjfdG9/LvsNl2f2bvJMnT3PO6K@7Tm/WIVx@/hl7Soxk25WFX3nF0NQ/P9EeheITbD4p7IHnHUk@8B9RtkjeQorqLTLU1S7rZNSLt3e4Ui3zoK7N60lWItrQLniL09oL5qYFoDo7vkaEB11WToKxnEfNBj7RqZUi8wW8ODi7rpsmi6rLOUtrnU81crR9divVaHp5sLfmuaz27anqnrxQs4OS28Xt8O7zjH6IUbitzhjKQvXa4nduqEQwwdFUNvzz7dNYbbz/1Txnc2l4pHzRq7Ogud7qiJEVwOwzhOUyN3uju@344LrS3@Q7jCbutzeeU7KW@CP5Ut02EzINAwLoe0wqJ1CX75Ys9rgqQ1Id8ryiC3PggBjzcZyBu5zxKyW5@rvnJ8F9cxtDi@PL4@zqsT1aJ7oFpYrBRFTBb5DAhm16sBfe6XEy@7Ey9fNfG6O/EaE1M2irN3@gh/J/1KQN4VkLdW5sOOCBy@njkv1i2t0KEWawjTdN6kw09G68WALgd0XcowR2h7A8XHqsaZ1oqp6lBDNKb4NK/mv1b4DTVgv6pFX8EbP@KMpkpVQWcu0dsKsZxfot80lv5uMN///l8) link.
